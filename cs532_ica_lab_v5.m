                         
     % = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =                           
% try generating two vectors with random scalings
% 
% v1 = [2 1];   v2 = [0.5 2];
% N = 200;
% 
% v1_total = v1'*(rand(1,N/2)-0.5);       % generate scalings of v1
% 
% v2_total = v2'*(rand(1,N/2)-0.5);       % generate scalings of v2
% 
% 
% 
% 
% v1_total = v1'*(randn(1,N/2));       % generate scalings of v1
% 
% v2_total = v2'*(randn(1,N/2));       % generate scalings of v2
% 
% 
% 
% v_total = [v1_total v2_total];  % combine two directions
% 
% 
% for i=1:length(v_total) %add noise
%     v_total(1,i) = v_total(1,i) + 0.05*randn;
%     v_total(2,i) = v_total(2,i) + 0.05*randn;
% end
% 
% 
% % plot both vectors
% figure; scatter(v_total(1,:)', v_total(2,:)','.');
% title('original data');


%%% Sine and Sawtooth  ==================================
clear all; close all;
xx = 0:0.05:10;      % 201 points

s1 = sin(xx);
s2 = rem(xx,0.25);

figure; plot(s1); hold on; plot(s2); hold off;
title('Original source signals');      % sin and sawtooth

S=[s1; s2];    % source signals

A = [0.3 0.7; ...
    0.8 0.3];          % mixing the source signals

X = A*S;               % 'measured' signal



% =======================================================


% gen data     ---- Chris' Gaussian data ------------
N=250;
A = [0.3,0.9;...
    0.8,0.1];

r = randn(1,N);
X = A*(randn(2,N).*[(r>=1/2);(r<1/2)]);
for i=1:length(X) %add noise
    X(1,i) = X(1,i);% + 0.05*randn;
    X(2,i) = X(2,i);% + 0.05*randn;
end


% plot data
figure; scatter(X(1,:)', X(2,:)','.');
title('Original data');


v_total=X; 


%%%%% ------------------ using Y- non gaussian data more densely generated
%v_total = X;
%%%%% ------------------ see at end of file

X_mean = mean(X,2);        % take mean along rows ('remove' cols). take mean of vector components, x1,x2

X_zerod = zeros(size(X));       
for i = 1:size(X,2)
    X_zerod(:,i) = X(:,i) - X_mean(:,1);      % zero mean each signal
end
% 
% figure; plot(X_zerod(1,:)); hold on; plot(X_zerod(2,:)); hold off;
% title('Original source signals');      % sin and sawtooth
%%% not plotting s, plotting x-the mixed signal

% plot both vectors
figure; scatter(X_zerod(1,:)', X_zerod(2,:)'); title('orig data zero mean');



% whiten
Cov = cov(X_zerod');    % covariance of the two variables (x1, x2)  
                                                                  
[U D V] = svd(Cov);     % svd to get eigen values and eigen vectors

d2 = diag(diag(D.^(-1/2)));          % diagonal matrix

X_zerod2 = (U*d2*U')*X_zerod;       % whiten data, zero variance for each component (x1,x2)

% plot whitened data
figure; scatter(X_zerod2(1,:)', X_zerod2(2,:)'); title('whitened X data'); 

% 
% d2= zeros(size(D));
% d2(eye(size(X_zerod,1))==1) = diag(D).^(-1/2);         % take D to the -1/2 power




%%%% ------ use the SVD to get the first principal component

[U2 S V2] = svd(v_total);

figure; scatter(v_total(1,:)', v_total(2,:)' ,'.');
hold on; plot([0 U2(1,1)], [0 U2(2,1)],'g', 'linewidth', 3); 
plot([0 U2(1,2)], [0 U2(2,2)],'g','linewidth', 3); hold off;
title('original data, PCA directions'); 




% *** Next *** try equation 45,46 for 'several unit' ICA
% let W = W / sqrt( norm(WW') )
% 
% repeat: W = 3/2 * W  -  1/2 * WW'W
% 
% W is the matrix of w1, w2, wn vectors transposed
% W = (w1, ... wn)'  w1 col vect is 1st row, w2 is 2nd row
%  

x = X_zerod2;       % x variable as in UCLA ICA ppr
 
% FAST ICA iteration --- for several units -------------------

W = rand(2,2)-0.5;     % initialize with 2 random vectors    -0.5 to 0.5 range

%%% for sine, sawtooth
%W = rand(2,2);         % initialize with 2 random vectors (positive here)


%%%% ---- iteration

diff_outer = 5;         % outer W+ magnitude difference
counter_out = 0;        % outer W+ counter

while (diff_outer > 1e-7)
    
% W_plus = W + gamma*( diag(-bi) + E(g(Wx)(Wx)')) *W
% 
% orthogonalize W with W=W/sqrt( norm(WW')),  W = 3/2W -1/2WW'W

% diag(-bi)
y = W*x;

g_y = tanh(2*y);                   % 2x16 matrix of data points

gy_mult = y.*g_y;                  % y*g_y component wise

gy_expect = mean(gy_mult,2);       % mean of components (remove cols), get row vect


diag_term = -1*diag(gy_expect);    % -bi term

 
term2_temp = zeros(size(x,2),size(diag_term(:),1));    % N x 4 matrix. N data pts
%term2 = g_y*y';                    % 2x2 matrix
for i = 1:size(x,2)   %loop through data pts
    term2_temp(i,:) = reshape(g_y(:,i)*y(:,i)', 1, 4);   % store each 2x2 matrix
end

% mean of matrix elements
mean_temp = mean(term2_temp,1);    % 1x4

mean_temp_mat = reshape(mean_temp, 2, 2);    % reshape back to 2x2 
 
term2 = mean_temp_mat; 
 

%gamma term/matrix

% bi is gy_expect
g_prime = 1-(tanh(2*y)).^2;              % 2xN

g_prime_expect = mean(g_prime,2);        % 2x1    'col vect that represents data'

term_g = gy_expect - g_prime_expect;
term_g = term_g.^(-1);               % -1 power   2x1 matrix
 
gamma = diag(term_g); 
 
 

% W_plus = W + gamma*( diag(-bi) + E(g(Wx)(Wx)')) *W
% 
% orthogonalize W with W=W/sqrt( norm(WW')),  W = 3/2W -1/2WW'W

W_plus = W + gamma*(diag_term + term2)*W;      % update W

W_plus = W_plus/sqrt(norm(W_plus*W_plus',2));  % normalize


W_old = W_plus;
diff =5;                      % inner magnitude difference

counter = 0;                  % inner counter

while (diff > 1e-7)           % orthogonalize W

    W_plus = (3/2)*W_plus - 0.5*(W_plus)*(W_plus')*W_plus;

    diff = norm(W_old-W_plus,2);

    W_old = W_plus;
    counter = counter+1;

end

if (counter_out == 1000)      % break out of loop for 1000 iterations
    break
end

diff_outer = norm(W-W_plus,2);


W = W_plus;          % update W to be orthogonalized W_plus;

counter_out = counter_out+1;

end  % outer iteration loop

                                      % didnt converge for 'Y' data with
                                      % 1-tanh()^2. converged with 2*(1-tan
                                      % didnt converge for 2*(), 
                                      % tried 3*(1-tan^2) and get conv. but
                                      % bad vector directions
                                      
                                      % tried 1-tan(3y)^2 and get conv but
                                      % bad vect directions
                                      
                                      % sometimes diff_outer =2.0 w
                                      % oscillates in sign
            
%%%% --- converged after counter_out= .. ;

% W  is constructed to be orthogonal    



% dewhiten data  (rotate 'projection' matrix)

W_dewhit = (U*(D.^(1/2))*U')*W;        % W_dewhit should be like W_i from FastICA

% A_dewhit = W_dewhit';    since W is orthogonal, inv(W) = W'


% plot dewhitened components
figure; scatter(X(1,:)', X(2,:)', '.'); title('Orig data, W+ iteration components');
hold on; plot([0 W_dewhit(1,1)], [0 W_dewhit(2,1)] ,'g', 'Linewidth', 3); hold off;    % red point
hold on; plot([0 W_dewhit(1,2)], [0 W_dewhit(2,2)], 'g', 'Linewidth', 3); hold off;     % green point





%%%%%
%%%
%%%
%%%
%%%%
%S_back = W_dewhit*X;           % this W is the estimated A, A holds the princip directions
                      % W_dewhit' = A;    since W is orthogonal, inv(W) = W'
                      % princip directions should be in columns, W -> w'
S_back = inv(W_dewhit)*X;     % take the inverse of 'A' (A has princip directions in rows, like our W found)

figure; plot(S_back(1,:)); hold on; plot(S_back(2,:)); hold off;
title('W+ Implemented ICA recovered signals');

%%%%%
%%%
%%%
%%%
%%%%


%%% --------- Fast ICA on vect v1,v2 whitened data


                           % examples in rows (signals, components in rows)
[icasig, A_i, W_i] = fastica(X); %, 'numOfIC', 2,  'maxNumIterations', 1000000);
                            %v_total or v_zerod2   -- 'approach', 'symm',

% plot both vectors - whitened
% figure; scatter(v_zerod2(1,:)', v_zerod2(2,:)'); title('whitened v_data');
% 
%                            % rows of A give directions
% hold on; plot([0 A_i(1,1)], [0 A_i(1,2)]); hold off;   % get princip component direction
% hold on; plot([0 A_i(2,1)], [0 A_i(2,2)]); hold off;   % get princip component direction


% plot data points and W - direction vectors   (Plot cols of A)
figure; scatter(X(1,:)', X(2,:)','.');
title('Orig data, FASTICA components');

                           % rows of W give directions, Cols of A for
                           % directions
hold on; plot([0 A_i(1,1)], [0 A_i(2,1)],'g', 'linewidth', 3); hold off;   % get princip component direction
hold on; plot([0 A_i(1,2)], [0 A_i(2,2)],'g', 'linewidth', 3); hold off;   % get princip component direction







%%%%%
%%%
%%%
%%%
%%%%

S_back_i = W_i*X;

figure; plot(S_back_i(1,:)); hold on; plot(S_back_i(2,:)); hold off;
title('FASTICA recovered signals');

%%%%%
%%%
%%%
%%%
%%%%





