                         
     % = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =                           
% try generating two vectors with random scalings

v1 = [2 1];   v2 = [1 4];

v1_total = v1;


for i = 1:20  %  extra/new points
    v1_new = 2*(rand-0.5)*v1; v1_total = [v1_total; v1_new];
end



v2_total = v2;

for i = 1:20  %  extra/new points
    v2_new = 2*(rand-0.5)*v2; v2_total = [v2_total; v2_new];
end


% plot both vectors
figure; scatter([v2_total(:,1); v1_total(:,1)], [v2_total(:,2); v1_total(:,2)]);
title('original data');

v_total = [v1_total; v2_total];  % combine two directions
v_total = v_total';              % place the 2 coordinates in rows


%%%%% ------------------ using Y- non gaussian data more densely generated
v_total = X;
%%%%% ------------------ see at end of file

v_mean = mean(v_total,2);        % take mean along rows ('remove' cols). take mean of vector components, x1,x2

% this mean direction is in between the two vectors

v_zerod = zeros(size(v_total));
for i = 1:size(v_total,2)
    v_zerod(:,i) = v_total(:,i) - v_mean(:,1);      % zero mean each component (x1,x2 ~variable)
end



% plot both vectors
figure; scatter(v_zerod(1,:)', v_zerod(2,:)'); title('orig data zero mean');



% whiten
Cov = cov(v_zerod');    % covariance of the two variables (x1, x2)    (variables in columns)
                                                                     %examples in rows

[U D V] = svd(Cov);     % svd for eigen values and eigen vectors

d2= zeros(size(D));
d2(eye(size(v_zerod,1))==1) = diag(D).^(-1/2);         % take D to the -1/2 power

v_zerod2 = (U*d2*U')*v_zerod;       % whiten data, zero variance for each component (x1,x2)

% plot whitened data
figure; scatter(v_zerod2(1,:)', v_zerod2(2,:)'); title('whitened v data'); 





%%%% ------ use the SVD to get the first principal component

[U2 S V2] = svd(v_total);

figure; scatter(v_total(1,:)', v_total(2,:)');
hold on; plot([0 U2(1,1)], [0 U2(2,1)],'g', 'linewidth', 2); 
plot([0 U2(1,2)], [0 U2(2,2)],'g','linewidth', 2); hold off;
title('original data, PCA direction'); 




% *** Next *** try equation 45,46 for 'several unit' ICA
% let W = W / sqrt( norm(WW') )
% 
% repeat: W = 3/2 * W  -  1/2 * WW'W
% 
% W is the matrix of w1, w2, wn vectors transposed
% W = (w1, ... wn)'  w1 col vect is 1st row, w2 is 2nd row
%  

x = v_zerod2;       % x variable as in UCLA ICA ppr
 
% FAST ICA iteration --- for several units -------------------

W = rand(2,2)-0.5;          % 2 random vectors    -0.5 to 0.5 range


%%%% ---- iteration

diff_outer = 5;         % outer W+ magnitude difference
counter_out = 0;        % outer W+ counter

while (diff_outer > 1e-5)
    

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

while (diff > 1e-5)           % orthogonalize W

    W_plus = (3/2)*W_plus - 0.5*(W_plus)*(W_plus')*W_plus;

    diff = norm(W_old-W_plus,2);

    W_old = W_plus;
    counter = counter+1;

end

if (counter_out == 10001)      % break out of loop for 5000 iterations
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
            
%%%% --- converged after counter_out= .. ;

% plot whitened data and whitened W directions
figure; scatter(v_zerod2(1,:)', v_zerod2(2,:)'); title('whitened v_data'); 
        
hold on; plot([0 W(1,1)], [0 W(1,2)]); hold off;    % green point
hold on; plot([0 W(2,1)], [0 W(2,2)]); hold off;     % red point

% W  is constructed to be orthogonal    



% dewhiten data  (rotate 'projection' matrix)

W_dewhit = (U*(D.^(1/2))*U')*W;        % W_dewhit should be like W_i from FastICA



% plot dewhitened components
figure; scatter(v_total(1,:)', v_total(2,:)'); title('orig data, W+ iteration');
hold on; plot([0 W_dewhit(1,1)], [0 W_dewhit(1,2)] ,'r', 'Linewidth', 3); hold off;    % red point
hold on; plot([0 W_dewhit(2,1)], [0 W_dewhit(2,2)], 'g', 'Linewidth', 3); hold off;     % green point





%%% --------- Fast ICA on vect v1,v2 whitened data


                           % examples in rows (signals, components in rows)
[icasig, A_i, W_i] = fastica(v_total, 'numOfIC', 2, 'maxNumIterations', 1000000);
                            %v_total or v_zerod2

% plot both vectors - whitened
% figure; scatter(v_zerod2(1,:)', v_zerod2(2,:)'); title('whitened v_data');
% 
%                            % rows of A give directions
% hold on; plot([0 A_i(1,1)], [0 A_i(1,2)]); hold off;   % get princip component direction
% hold on; plot([0 A_i(2,1)], [0 A_i(2,2)]); hold off;   % get princip component direction


% plot data points and W - direction vectors
figure; scatter(v_total(1,:)', v_total(2,:)');
title('original data FASTICA');

                           % rows of W give directions
hold on; plot([0 W_i(1,1)], [0 W_i(1,2)], 'linewidth', 3); hold off;   % get princip component direction
hold on; plot([0 W_i(2,1)], [0 W_i(2,2)], 'linewidth', 3); hold off;   % get princip component direction




%%%% try with denser gaussian created set


% ica data
n=500;       % some cases of 100 data did not converge

 
A = randn(2,2);         % 2 directions

 
% gaussian data
X = A*randn(2,n);       % scale in two directions

 
% non-gaussian data
r = randn(1,n);
Y = A*(randn(2,n).*[(r>=1/2);(r<1/2)]);     % use Y as x in iteration above

 
figure(1)
subplot(1,2,1)
scatter(X(1,:),X(2,:),'.')
axis('square')
title('gaussian data')
subplot(1,2,2)
scatter(Y(1,:),Y(2,:),'.')
axis('square')
title('nongaussian data')





% other data generation:


n=1000;
A = [0.3,0.9;0.8,0.1];
r = randn(1,n);
X = A*(randn(2,n).*[(r>=1/2);(r<1/2)]);
for i=1:length(X) %add noise
    X(1,i) = X(1,i) + 0.05*randn;
    X(2,i) = X(2,i) + 0.05*randn;
end          
                                                   % v_total
                                          % use X as x in iteration above
                                          
                                          

                                          
%%% try sine and sawtooth       (harder to visualize in 2-coordinate space)
                                          
