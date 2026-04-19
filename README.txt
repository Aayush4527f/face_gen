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