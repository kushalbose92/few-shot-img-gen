import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Model defintion
class ImageGenerator(nn.Module):
    def __init__(self, latent_dims, ways, shot):
        super(ImageGenerator, self).__init__()
        self.latent_dims = latent_dims
        self.ways = ways
        self.shot = shot
        in_channels = 1


        # Encoder
        self.enc_layer_1 = nn.Sequential(
                                    nn.Conv2d(in_channels, 64, 4, stride = 2, padding = 1),
                                    nn.BatchNorm2d(64, track_running_stats = False),
                                    nn.LeakyReLU(negative_slope = 0.2))

        self.enc_layer_2 = nn.Sequential(
                                    nn.Conv2d(64, 128, 4, stride = 2, padding = 1),
                                    nn.BatchNorm2d(128, track_running_stats = False),
                                    nn.LeakyReLU(negative_slope = 0.2))

        # Decoder
        self.dec_layer_1 = nn.Sequential(
                                    nn.ConvTranspose2d(128, 64, 4, stride = 2, padding = 1),
                                    nn.BatchNorm2d(64, track_running_stats = False),
                                    nn.LeakyReLU(negative_slope = 0.2))

        self.dec_layer_2 = nn.Sequential(
                                    nn.ConvTranspose2d(64, in_channels, 4, stride = 2, padding = 1),
                                    nn.BatchNorm2d(in_channels, track_running_stats = False))


        # Mapping to latent space
        self.layer_latent = nn.Sequential(nn.Linear(128*7*7, self.latent_dims), nn.BatchNorm1d(self.latent_dims, track_running_stats = False))

        # Radius generator for 1-way-k-shot problem else comment this hidden layer
        if self.ways == 1:
            self.layer_radius = nn.Sequential(nn.Linear(latent_dims, latent_dims))

        # Mapping from latent space to embedded space
        self.fc = nn.Sequential(nn.Linear(self.latent_dims, 128*7*7), nn.BatchNorm1d(128*7*7, track_running_stats = False))


    # Class prototype estimation
    def class_prototypes(self, x_latent):
        x_proto = torch.zeros(self.ways, self.latent_dims).to(device)
        num_points = x_latent.size(0)

        for i in range(self.ways):

            class_avg = torch.zeros(1, self.latent_dims).to(device)
            for j in range(self.shot):

                class_avg += x_latent[i*self.shot + j]
            class_avg /= self.shot
            x_proto[i] = class_avg

        return x_proto


    # Neighborhood radius generation
    def radius_generation(self, x_latent):

        num_points = x_latent.size(0)
        latent_dims = x_latent.size(1)
        feat_diff = torch.zeros(num_points, num_points, latent_dims).to(device)
        attn = torch.zeros(num_points, num_points).to(device)
        radius_vector = torch.zeros(num_points, latent_dims).to(device)

        attn = torch.matmul(x_latent, torch.t(x_latent)).to(device)
        attn = F.softmax(attn, dim = 1)

        for i in range(num_points):
            for j in range(num_points):

                feat_diff[i][j] = torch.sub(x_latent[i], x_latent[j])

        for i in range(num_points):
            radius_vector[i] = torch.matmul(attn[i].unsqueeze(0), feat_diff[i])

        return radius_vector

    # Encoding of the input image
    def encoder(self, input):
        x1 = self.enc_layer_1(input)
        x2 = self.enc_layer_2(x1)
        x_flatten = x2.view(x2.size(0), -1)
        x_latent = torch.sigmoid(self.layer_latent(x_flatten))
        if self.ways > 1:
            x_proto = self.class_prototypes(x_latent)
        else:
            x_proto = torch.mean(x_latent, dim = 0).unsqueeze(0)

        return x_proto, x_latent

    # Decoding of the encoded image set
    def decoder(self, input):
        x_embed = self.fc(input)
        x = x_embed.view(-1, 128, 7, 7)
        x1 = self.dec_layer_1(x)
        x2 = self.dec_layer_2(x1)
        x2 = torch.sigmoid(x2)
        return x2

    # Forward propagation of the model
    def forward(self, input, gamma):
        x_proto, x_latent = self.encoder(input)

        if self.ways > 1:
            x_radius = self.radius_generation(x_proto)
        else:
            x_radius = torch.sigmoid(self.layer_radius(x_proto))

        repeated_radius = torch.zeros(self.ways*self.shot, self.latent_dims).to(device)
        for k in range(self.ways * self.shot):
            repeated_radius[k] = x_radius[int(k / self.shot)]

        x_recon = torch.add(x_latent, repeated_radius * gamma)
        recon_images = self.decoder(x_recon)

        return recon_images, x_proto, x_radius, x_latent
