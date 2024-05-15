from torch import nn
import functools

class Discriminator(nn.Module):
    def __init__(self, input_nc, nf=64, n_layers=3, num_D=1):
        super(Discriminator, self).__init__()
        self.num_D = num_D
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False,
                                        track_running_stats=False)

        for i in range(num_D):
            layers = self.get_layers(input_nc, norm_layer, nf, n_layers)
            module_name = "model" if i == 0 else f"model_{i}"
            self.add_module(module_name, nn.Sequential(*layers))
            nf = int(round(nf / 2))

        if num_D > 1:
            self.down = nn.AvgPool2d(3, stride=2, padding=[
                                     1, 1], count_include_pad=False)
                

    def get_layers(self, input_nc, norm_layer, nf=64, n_layers=3):
        layers = [nn.Conv2d(input_nc, nf, kernel_size=4, stride=2, padding=1), 
                  nn.LeakyReLU(0.2, True)]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers+1):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            s = 2 if n != n_layers else 1            
            layers += [
                nn.Conv2d(nf * nf_mult_prev, nf * nf_mult,
                          kernel_size=4, stride=s, padding=1),
                norm_layer(nf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        layers += [nn.Conv2d(nf * nf_mult, 1,
                               kernel_size=4, padding=1)]

        return layers

    def forward(self, input):
        if self.num_D == 1:
            return getattr(self, "model")(input)
        else:
            results = []
            for i in range(self.num_D):
                module_name = "model" if i == 0 else f"model_{i}"
                model = getattr(self, module_name)
                results.append(model(input))
                input = self.down(input, kernel_size=3, stride=2, padding=[1, 1], count_include_pad=False)
            return results
    
# if __name__ == "__main__":
#     print(Discriminator(3, 64, 3, 1))