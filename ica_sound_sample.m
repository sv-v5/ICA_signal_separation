% ICA on audio sample

%audioinfo('ica_sound')       % --- 2007 version of matlab doesn't have audio write and read

%audiowrite('out_file.mp3',randn(100,1),0.1);

t = 0:0.1:2;

x = sin(t) + cos(10*t);
x2 = x+0.5*randn(size(x));

figure('position', [5 550 400 500]); plot(1:size(x,2),x2);

output = fastica([x;x2], 'numOfIc', 2);
[A W] = fastica([x;x2]);

figure('position', [5 10 400 500]); plot(1:size(output,2),output);

size(output);




%%%% ---- couldn't get 2 or more components from signal x



