from stable_diffusion_jax import AutoencoderKL
from stable_diffusion_jax.modeling_vae import AutoencoderKLModule
import jax.numpy as jnp
import random 
import jax
class TemporalAutoEncoderKLModule (AutoencoderKLModule):

    def decode(self, latents, pixel_values2, deterministic: bool = True):
        # Concatenate latents and the second image along the channel dimension
        combined_input = jnp.concatenate([latents, pixel_values2], axis=-1)

        hidden_states = self.post_quant_conv(combined_input)
        hidden_states = self.decoder(hidden_states, deterministic=deterministic)
        return hidden_states
    
    def __call__(self, sample1, sample2=None, sample_posterior=False, deterministic: bool = True):
        posterior1 = self.encode(sample1, deterministic=deterministic)
        if sample_posterior:
            rng = self.make_rng("gaussian")
            hidden_states1 = posterior1.sample(rng)
        else:
            hidden_states1 = posterior1.mode()

        if sample2 is not None:
            posterior2 = self.encode(sample2, deterministic=deterministic)
            hidden_states2 = posterior2.mode()  # Or sample, depending on your needs
        else:
            hidden_states2 = hidden_states1  # Or some other default behavior

        # Decode using both the hidden states and the second image
        hidden_states = self.decode(hidden_states1, hidden_states2)
        return hidden_states, posterior1
    
    def replace_with_params(self, params):
        # Create a new model with the same configuration and dtype
        new_model = TemporalAutoEncoderKL(self.config, self.dtype)

        # Create a mock input to initialize the model
        mock_input = jnp.ones((1, self.config.sample_size, self.config.sample_size, self.config.in_channels), dtype=self.dtype)

        # Initialize the model
        _, initial_params = new_model.init_by_shape(random.PRNGKey(0), [(mock_input.shape, self.dtype)])

        # Replace the initial parameters with the given parameters
        new_model_params = {**initial_params, **params}
        
        return new_model, new_model_params
    

    


class TemporalAutoEncoderKL(AutoencoderKL):
    module_class = TemporalAutoEncoderKLModule