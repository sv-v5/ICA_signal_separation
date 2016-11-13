P=path;
path(P,'C:\Users\Christopher\Desktop\Good Code\FastICA_25');

[S1_orig, sample_rate1_orig] = audioread('S1_orig.wav');
[S2_orig, sample_rate2_orig] = audioread('S2_orig.wav');
[S1_S2_mixed, sample_rate_mixed_orig] = audioread('S1_S2_mixed.wav');

figure
% original
subplot(3,2,1);
plot(S1_orig','g');
subplot(3,2,2);
plot(S2_orig','b');

% mixed
subplot(3,2,3);
plot(S1_S2_mixed(:,1)','r');
subplot(3,2,4);
plot(S1_S2_mixed(:,2)','c');

% estimates
[ICA_est, A_est, W_est] = fastica(S1_S2_mixed','verbose', 'off');
subplot(3,2,5);
plot(ICA_est(1,:)','g');
subplot(3,2,6);
plot(ICA_est(2,:)','b');

% plot A vector directions
A_orig = [1 0.7; 0.3 1];
figure
subplot(1,2,1)
plot([0 A_orig(1,1)], [0 A_orig(1,2)],'b', [0 A_orig(2,1)], [0 A_orig(2,2)],'g');
subplot(1,2,2)
plot([0 A_est(1,1)], [0 A_est(1,2)],'b', [0 A_est(2,1)], [0 A_est(2,2)],'g');



% play audio files
sound1 = audioplayer(S1_orig,sample_rate1_orig);
sound2 = audioplayer(S2_orig,sample_rate2_orig);

play(sound1);
play(sound2);


sound3 = audioplayer(S1_S2_mixed,sample_rate_mixed_orig);

play(sound3);      % mixed sound


% ICA unmixed
sound4=audioplayer(ICA_est(1,:),sample_rate1_orig);
sound5=audioplayer(ICA_est(2,:),sample_rate2_orig);



play(sound4);
play(sound5);      % separated, but volume was increased



figure; plot(S1_S2_mixed);



figure; plot(ICA_est(1,:));



% scale amplitude
sound6=audioplayer(ICA_est(1,:)/max(abs(ICA_est(:))),sample_rate1_orig);

play(sound6);     % softer volume


figure; plot(ICA_est(1,:)/max(abs(ICA_est(:))));



sound7=audioplayer(ICA_est(2,:)/max(abs(ICA_est(:))),sample_rate2_orig);

play(sound7);     % softer volume


figure; plot(ICA_est(2,:)/max((abs(ICA_est(:)))));

