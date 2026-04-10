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
