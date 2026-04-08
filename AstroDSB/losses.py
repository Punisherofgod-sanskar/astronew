from .constraints import smoothness_loss, mass_conservation_loss

class DensityLoss:
    def __init__(self, w_smooth=0.025, w_mass=0.005):
        self.w_smooth = w_smooth
        self.w_mass = w_mass

    def __call__(self, rho):
        loss = 0.0
        loss += self.w_smooth * smoothness_loss(rho)
        loss += self.w_mass * mass_conservation_loss(rho)
        return loss