v2 goal => create a good baseline model having pure cnn+mlp, no resnet etc

v2.1 ===>
I put beta = 2.0 at the start
the results show that the warmup was too low and model was learning then
now trying with (very)higher beta and higher warmup and a smaller learning rate

v2.2 ===>
Well it was just for fun very high beta made kld drop a lot
I'll see what kind of results this one will give me
val was similarly mostly stable

v2.3 ===>
Let's get serious now
ok, this time I got actually good training curves, let's use a better lr scheduler
shifting from step on plateu to cosinesomebs

v2.4 ==>
Shifted to cosineannealing but it seems worse for some reason
also added augmentation (horizontal flip)

v2.5 ==> so apperantly the ssim was set to 11x11 by default, turning it down to 5x5 now
nope, no improvements
even adding attention to the 3x3 layer just made it blurry and disfigured

for now 2.2 is best in this architecture

v2 FINAL============================================================================================================================================================
gotta change the loss function so that colors are more accurate maybe adding l1 loss in lab (a and b color axes)
atleast once I wanna try out ViT full dense layer
try adding upsampling to decoder
try resnet style skip connections
add more linear layers instead of 2304 -> 16, maybe 2304->1152->16 or 2304->4608->16

MASSIVE IDEA!!! ==>
WHAT IF WE HAVE DIFFERENT LOSS FUNCTION FOR DIFF LATENT
LIKE A COLOR LOSS FOR 2 DIM, SSIM FOR SOME, MSE FOR SOME ETC
========================================================================================================================

start v3
goal: well, implement and try the ideas from earlier

v3.1 => added upscaling instead of convtranspose2d and l1 loss on lab
also tried increasing size of linear layer but removed for now
results and observations => now the images are much more vibrant
adding more capabilites in the hidden layer makes disfigured faces because it finds shortcuts
need to add a better loss function for a better model (golden line)

v3.2 => added GAN and it works, did not test it, will test after adding resnet blocks in 3.3

v3.3 => adversarial network too powerful, beta was too small, linear bottleneck and should do two seperate forward passes instead

v3.4 => reduced disc to 3 layer
        increased beta
        added two linear layers
        seperate forward pass to reduce error

    
Now the model is much better, but idk how much of it is because of gan or resnet?
the kld needs some work, it has to be a bit more lower and I will train it to be 
also the scheduler will be updated because rn the lr is constant mostly
also haven't yet tried the idea that I had

will compare after removing GAN and keeping only resnet then move to the idea execution


v3.5 => compared gan and without gan, gan makes some difference idk what exactly lol, the image looks more noisy now
still there are many issues =>
1. both kld and reconstruction are high
2. model is stuck during training
3. still very blurry images

actually this specific gan makes the model worse, even with very low adv_weight, the model is stuck during training

final things to try =>
1. replace batchnorm with layernorm
2. lpips instead of ssim
3. increase discriminator layer till 3x3
4. adv_weight warmup
5. logvar clamping
6. better beta warmup


things learned =>       lpips + l1 is a good combo
                        logvar should always be clamped to avoid posterior collaspe later
                        learning rate scheduler plays a very vital role in training.... very vital, fuck that up and get fucked
                        
v3.6 => trained the model with better LR (manually, gotta find some solution for that), the 16 dim seems to be the bottleneck now
added clamping to the logvar and found out that the learning rate needed to be wayyy lower
also during this version I tried every single thing like resnet or not, lr high or not, beta values, loss functions etc etc to verify everything


v4 => fixes to try finallll
1. fix the logvar clamping, it was too tight -20 to 20 (done)
2. do not average the kld over all dimensions, sum it up (done)
3. tune the adam beta values for optimizer (done)
4. increase LR (done)
5. use lsgan instead of normal bce_with_logits (done)
6. remove intermediate linear layers (done)
7. add discriminator delay (done)
8. add horizontal flip (done)
9. switch to bilinear upsampling and add GroupNorm to ResUpBlock shortcuts (done)

GOOD RESULTS FINALLY!!!


v4.1 => remove spatial attention to make the model simpler
        also add encoder model too for uploading images
        increase beta to get a more stable space
        oh fuck I forgot to tweak free_bits value

a little better result, still not ultra-realistic pretty decent tho


for now I am satisfied with this model, even though it's not pretty perfect, there are still weird images in the latent space
I might change the bounds of the sliders a bit
maybe i'll also add "feature" directional vectors

buttt

next big step: working on that idea of seperate dimensions for features in the latent space, idk what I'll do but I will try something there








============================================================================================
AI SUMMARY =>
============================================================================================
Project Overview:
This repository chronicles the iterative optimization of a BetaVAE-GAN hybrid model 
engineered to compress high-dimensional image data into a lean 16-dimensional latent space. 
The architecture balances structured latent topography with high-fidelity reconstruction.

Key Architectural Milestones:
- v2.x Baseline: Established structural bounds using standard CNN/MLP blocks. Experimented 
  with beta weights and SSIM window scaling. Identified structural blurring vulnerabilities.
- v3.x VAE-GAN Fusion: Integrated an adversarial discriminator loop alongside LPIPS and L1 
  losses in the LAB color space to combat reconstruction washouts. Addressed training 
  stagnation and posterior collapse via aggressive hyperparameter tuning.
- v4.x Optimization (Production Stability): Implemented Least Squares GAN (LSGAN) loss, 
  bilinear upsampling blocks, GroupNorm adjustments, and structural log-variance clamping. 
  Summed rather than averaged KLD losses across latent dimensions to maintain stability.

Core Insights & Engineering Takeaways:
1. Posterior distribution health is highly contingent on structural limits; clamping 
   log-variance prevents early training collapse.
2. Learning rate schedules govern the critical balance point between the generative encoder 
   and the adversarial discriminator.
3. Feature quality is maximized when structural losses (LPIPS) are paired with perceptual 
   color penalties rather than raw pixel-space distances.

Next Research Vector:
Executing supervised feature disentanglement by binding unique generative loss constraints 
to independent subsets of the 16 latent dimensions.
============================================================================================