"""
This class represents an image buffer designed to store previously generated images. 
By maintaining a history of generated images, the buffer facilitates updating discriminators 
with a diverse set of images from the past, rather than relying solely on the latest images 
produced by the generators
"""
import random
import torch

class ImagePool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:  
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images

        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if len(self.images) < self.pool_size:
                self.images.append(image)
                return_images.append(image)
            else:
                if random.random() > 0.5:
                    idx = random.randint(0, self.pool_size - 1)
                    tmp = self.images[idx].clone()
                    self.images[idx] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)

        return torch.cat(return_images, 0)
