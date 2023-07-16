import torch
import argparse
from ddpm import UNet
from tqdm import tqdm
import os 
from torchvision.utils import save_image  
import uuid

class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=64, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n):
        # logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        # x = (x.clamp(-1, 1) + 1) / 2 #scale to [0,1]
        # x = (x * 255).type(torch.uint8)
        return x
    

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Diffusion Image Sampling")
    parser.add_argument("--noise-steps", type=int, default=1000, help="Number of noise steps")
    parser.add_argument("--beta-start", type=float, default=1e-4, help="Starting beta value")
    parser.add_argument("--beta-end", type=float, default=0.02, help="Ending beta value")
    parser.add_argument("--img-size", type=int, default=64, help="Image size")
    parser.add_argument("--device", type=str, default="cuda", help="Device ('cuda' or 'cpu')")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")


    parser.add_argument("--gen-path", type=str, default="")
    parser.add_argument("--weight-path", type=str, default="")
    parser.add_argument("--sample-size", type=int, default=1000)
    args = parser.parse_args()

    # Create an instance of the Diffusion class
    diffusion = Diffusion(noise_steps=args.noise_steps,
                          beta_start=args.beta_start,
                          beta_end=args.beta_end,
                          img_size=args.img_size,
                          device=args.device)

    model = UNet(device=args.device).to(args.device)  # Replace with your generative model

    index_weight = 1

    model.load_state_dict(torch.load(f'{args.weight_path}',map_location=torch.device(args.device)))


    # Set the model to evaluation mode
    model.eval()

    # Create a folder for generated images if it doesn't exist
    os.makedirs(args.gen_path, exist_ok=True)

    # Generate images
    with torch.no_grad():
        for i in range(0, args.sample_size, args.batch_size):

            x = diffusion.sample(model, n=args.batch_size).type(dtype=torch.float32)

            for j in range(args.batch_size):
                image_id = str(uuid.uuid4())
                save_image(x[j], f"{args.gen_path}/{image_id}.png", normalize=True)
        
    # model = None
    # gc.collect() 

    with torch.no_grad():
        del model
        torch.cuda.empty_cache()

    print(f"subprocess done.")