def column_density(rho):
    # integrate along z-axis (assume dim=2 or 3 depending on your tensor)
    # assuming shape: [B, 1, Z, H, W]
    return rho.sum(dim=2)   # -> [B, 1, H, W]


def column_density_loss(rho, NH_obs):
    NH_pred = column_density(rho)
    return ((NH_pred - NH_obs)**2).mean()

def velocity_dispersion(vz):
    # vz shape: [B, 1, Z, H, W]

    mean_v = vz.mean(dim=2, keepdim=True)
    mean_v2 = (vz**2).mean(dim=2, keepdim=True)

    sigma2 = mean_v2 - mean_v**2

    return sigma2.squeeze(2)   # -> [B, 1, H, W]


def velocity_dispersion_loss(vz, sigma_obs):
    sigma_pred = velocity_dispersion(vz)
    return ((sigma_pred - sigma_obs)**2).mean()


class MHDLoss:
    def __init__(self,
                 wdiv=0.1,
                 wpress=0.001,
                 wcol=0.1,
                 wvel=0.1):

        self.wdiv = wdiv
        self.wpress = wpress
        self.wcol = wcol
        self.wvel = wvel

    def __call__(self, x0, NH_obs=None, sigma_obs=None):

        rho = x0[:, 0:1]        # [B,1,Z,H,W]
        B   = x0[:, 1:3]
        vz  = x0[:, 5:6]        # assuming vz exists

        loss = 0.0

        # magnetic
        loss += self.wdiv   * divergence_loss(B)
        loss += self.wpress * magnetic_pressure_loss(rho, B)

        # density-based
        if NH_obs is not None:
            loss += self.wcol * column_density_loss(rho, NH_obs)

        if sigma_obs is not None:
            loss += self.wvel * velocity_dispersion_loss(vz, sigma_obs)

        return loss