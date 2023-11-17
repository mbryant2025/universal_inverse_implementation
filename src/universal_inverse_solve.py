import numpy as np
import torch


def universal_inverse_solver(M, M_T, x_c, denoiser, sigma_0=1, sigma_L=0.01, h_0=0.01, beta=0.01):

    # M and M_T depend on the task we are performing, so input them as arguments depending on the task (ex infill)

    t = 1

    # e is a matrix of 1's the shape of x_c
    e = np.ones(x_c.shape)

    # Draw y_0 from N(0.5 * (I - M^T M) e + M * x_c , sigma_0^2 * I)
    y_t = torch.distributions.normal.Normal(0.5 * (torch.eye(x_c.shape[0]) - torch.matmul(M_T, M)) * e + torch.matmul(M, x_c), sigma_0 ** 2 * torch.eye(x_c.shape[0])).sample()

    sigma_t = sigma_0

    while sigma_t <= sigma_L:

        # Step size
        h_t = (h_0 * t) / (1 + h_0 * (t - 1))

        # Denoised image
        #  f(y_t) = x^ (y) - y
        # Denoised is the output of the denoiser - the original image

        f = denoiser(y_t) - y_t

        # d_t = (I - M M^T) f(y_t) + M (x_c - M^T y_t)
        d_t = (torch.eye(x_c.shape[0]) - torch.matmul(M, M_T)) * f + torch.matmul(M, x_c - torch.matmul(M_T, y_t))

        # sigma_t = sqrt(abs(d_t)^2/N)
        # N is the number of pixels in the image
        sigma_t = torch.sqrt(torch.abs(d_t)**2 / x_c.shape[0] * x_c.shape[1])

        # gamma_t = sqrt((1 beta * h_t)^2 - (1 - h_t)^2) * sigma_t^2)
        gamma_t = torch.sqrt((1 - beta * h_t) ** 2 - (1 - h_t) ** 2) * sigma_t ** 2

        # Draw z_t from N(0, I)
        z_t = torch.distributions.normal.Normal(0, 1).sample()

        # y_t+1 = y_t + h_t * d_t + gamma_t * z_t
        y_t = y_t + h_t * d_t + gamma_t * z_t

        t += 1

    return y_t

