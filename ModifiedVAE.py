from stable_diffusion_jax import AutoencoderKL
import jax.numpy as jnp

class TemporalAutoEncoderKL (AutoencoderKL):

    def decode(self, latents, pixel_values2, deterministic: bool = True):
        # Concatenate latents and the second image along the channel dimension
        combined_input = jnp.concatenate([latents, pixel_values2], axis=-1)

        hidden_states = self.post_quant_conv(combined_input)
        hidden_states = self.decoder(hidden_states, deterministic=deterministic)
        return hidden_states
    
    def __call__(self, sample1, sample2, sample_posterior=False, deterministic: bool = True):
        posterior = self.encode(sample1, deterministic=deterministic)
        if sample_posterior:
            rng = self.make_rng("gaussian")
            hidden_states = posterior.sample(rng)
        else:
            hidden_states = posterior.mode()
        # Decode using both the hidden states and the second image
        hidden_states = self.decode(hidden_states, sample2)
        return hidden_states, posterior