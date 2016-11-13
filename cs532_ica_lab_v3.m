%%%% ICA implementation using FASTICA algorithm


x = [1 7 2 9 12 5 7 21 3 7 9 4 13 5 6 2 3 18 23 2 5 12 22]; x=x';
   
% sine wave
xx = 0:0.5:12;

x = sin(xx); x=x';

figure; plot(1:size(x,1),x); title('x zero mean');

% subtract the mean (zero data)
x = x - mean(x);


% whiten data using SVD

Cov = x*x';

[U D V] = svd(Cov);     % svd for eigen values and eigen vectors

D(D<1e-15)=0;           % keep eigen values above 1e-15 (as for x above)

d2= zeros(size(D));
d2(eye(size(x,1))==1) = diag(D).^(-1/2);         % take D to the -1/2 power

d2(d2==Inf)=0;            % keep only the first eigen value (singular value). the rest were close to 0
                          % keep the first 14 eigen values <1e-15

x2 = (U*d2*U')*x;       % whiten data


figure; plot(1:size(x2,1),x2); title('whitened x');            % reduces scale of x data

% hold on; plot(1:size(x,1),x,'r');       




% FAST ICA iteration

w = rand(size(x))-0.5;      % 23x1 vector  (-.5 to .5 range)

% own random vector
w = [1; 2; 4; 2; 2; 4; 5; 1; 2; 5; 3; 1; 3; 9; 7; 7; 6; 2; 1; 4; 8; 4; 8; 3; 11];
w = w -6;
w = w/6;

w_old = w;

diff = 5;
counter = 0;

diff_norm = zeros(500,1);
% x = x/7;  % if x is too small (scaled) w does not converge
% try a1 constant to be = 2 (not 1 as before)

while(diff > 1e-5)   % loop until diff is under tolerance level
    w_plus = x2*tanh(2*w'*x2) - w*(1-(tanh(2*w'*x2))^2);    % x or x2. ---converges for a1=2
    w = w_plus/norm(w_plus);             % normalize
    
    diff = norm(w-w_old);
    diff_norm(counter+1) = diff;
    
    %     if (norm(w_old - (-1*w)) <1e-5)
%         break;  % if w_old and w oscillate signs
%     end

    w_old = w;            % set w_old as w for next iteration
    counter=counter+1;
    
    if counter==500
        break;
    end
end


figure; plot(1:size(w,1),w); title('w vector, principal direction');
hold on; plot(1:size(x2,1),x2,'r');
hold off;

% this recovers the x data, but smaller scaled
%%% --- did not work well with whitened data (x2, with 1 singular value)
%%%%%%% ----- works now with a1 = 2 (had a1 constant, in g function, a1=1)

%%% --- does not work with whitened data now (kept 14 singular values)
%%%%%%% ---- increased the scale of the 'input' w'*x -> 2*w'*x. works



% look at tanh() and 1-tanh()^2
xx=0:0.5:5;
figure; plot(1:size(xx,2), tanh(xx)); title('tanh()');

figure; plot(1:size(xx,2), 1-(tanh(xx).^2)); title('1-tanh()^2');

%%%%%
%%%%% NOTE: was using logcosh and tanh, needed to use tanh and tanh'
%%%%%

% w oscillates to -w and back after 2 iterations (for a1=1)



 %%%
%%% --- try 5 vectors of same direction as x, but scaled randomly
 %%%

xs_1 = (4*(rand(1)-0.5))*x2;              % -2 to 2 scaling
xs_2 = (4*(rand(1)-0.5))*x2;
xs_3 = (4*(rand(1)-0.5))*x2;
xs_4 = (4*(rand(1)-0.5))*x2;
xs_5 = (4*(rand(1)-0.5))*x2;

figure; plot(1:size(x2,1),x2,'k'); title('scaling the original x2 vector');
hold on; 
plot(1:size(x,1),xs_1,'r');
plot(1:size(x,1),xs_2,'c');
plot(1:size(x,1),xs_3,'g');
plot(1:size(x,1),xs_4,'b');
plot(1:size(x,1),xs_5,'m');
hold off;

% for calculation of w, take the mean of each vector component as the
% expectation value of that component
data_x = [x2 xs_1 xs_2 xs_3 xs_4 xs_5];

data_x_mean = mean(data_x,2);             % mean vector ~E(x)

figure; plot(1:size(data_x_mean,1),data_x_mean); title('x_mean_1');

x_mean_1 = data_x_mean - mean(data_x_mean);       % new x vector, centered at zero

%x_mean_1 = -1*x_mean_1;      % multiply to get sign of original x2


% whiten x_mean_1

Cov2 = x_mean_1*x_mean_1';

[U D V] = svd(Cov2);     % svd for eigen values and eigen vectors

%D(D<1e-15)=0;          % keep eigen values above 1e-15 (as for x above)


d2= zeros(size(D));
d2(eye(size(x,1))==1) = diag(D).^(-1/2);         % take D to the -1/2 power

d2(d2==Inf)=0;         % keep only the first eigen value (singular value). the rest were close to 0



x_mean_2 = (U*d2*U')*x_mean_1;       % whiten x_mean_1

figure; plot(1:size(x_mean_2,1),x_mean_2); title('x_mean_2');
% hold on; plot(1:size(x2,1),x2,'r'); hold off;
%%% --- x_mean_1 becomes x2, the same vector direction whitened results in
%%% the same vector


% FAST ICA iteration

w = rand(size(x_mean_2))-0.5;      % 23x1 vector   -.5 to .5 range. change scale for whitened data
w_old = w;      % 1*(rand(size(x_mean_2))-0.5)

diff = 5;
counter = 0;
diff_norm = zeros(500,1);

while(diff > 1e-5)   % loop untill diff is under tolerance level
    w_plus = x_mean_2*tanh(2*w'*x_mean_2) - w*(1-(tanh(2*w'*x_mean_2))^2);
    w = w_plus/norm(w_plus);             % normalize
    
    diff = norm(w-w_old);
    diff_norm(counter+1) = diff;
    
    w_old = w;            % set w_old as w for next iteration
    counter = counter+1;
    
    if counter==500
        break
    end
end


figure; plot(1:size(w,1),w); title('w vector, principal direction');

% x works, recover the 'principal direction'
% x_whitened works, with a1=2, increased constant in 'g' function
                               
                               
% scaling of x and initial random w influence the iteration, recovery of x





                               
                               
                               
     % = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =                           
% try generating two vectors with random scalings

v1 = [2 1];   v2 = [1 4];

v1_total = v1;

figure; scatter(v1_total(:,1), v1_total(:,2));

%v1_new = rand*v1; v1_total = [v1_total; v1_new]; 

for i = 1:7  % 7 extra/new points
    v1_new = 2*(rand-0.5)*v1; v1_total = [v1_total; v1_new];
end




v2_total = v2;

for i = 1:7  % 7 extra/new points
    v2_new = 2*(rand-0.5)*v2; v2_total = [v2_total; v2_new];
end

% figure; scatter(v2_total(:,1), v2_total(:,2));


% plot both vectors
figure; scatter([v2_total(:,1); v1_total(:,1)], [v2_total(:,2); v1_total(:,2)]);
title('original data');

v_total = [v1_total; v2_total];  % combine two directions
v_total = v_total';              % place the 2 coordinates in rows


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

%D(D<1e-15)=0;           % keep eigen values above 1e-15 (as for x above)

d2= zeros(size(D));
d2(eye(size(v_zerod,1))==1) = diag(D).^(-1/2);         % take D to the -1/2 power

d2(d2==Inf)=0;            % keep only the first eigen value (singular value). the rest were close to 0
                          % keep the first 14 eigen values <1e-15

v_zerod2 = (U*d2*U')*v_zerod;       % whiten data, zero variance for each component (x1,x2)


figure; scatter(v_zerod2(1,:)', v_zerod2(2,:)'); title('whitened v_data'); 


% FAST ICA iteration --- for ONE DATA POINT (1-unit as in UCLA ppr)

w = rand(size(v_mean2))-0.5;      % 23x1 vector   -.5 to .5 range. change scale for whitened data
w_old = w;      % 1*(rand(size(x_mean_2))-0.5)

diff = 5;
counter = 0;
diff_norm = zeros(500,1);

while(diff > 1e-5)   % loop untill diff is under tolerance level
    w_plus = v_mean2*tanh(2*w'*v_mean2) - w*(1-(tanh(2*w'*v_mean2))^2);
    w = w_plus/norm(w_plus);             % normalize
    
    diff = norm(w-w_old);
    diff_norm(counter+1) = diff;
    
    w_old = w;            % set w_old as w for next iteration
    counter = counter+1;
    
    if counter==500
        break
    end
end


figure; plot(1:size(w,1),w); title('w vector, principal direction');


%%% - recovered v_mean [0.707, -0.707]
% can add back the means of vect component1 and component2
% [.7706, 1.4909] - but these are the non whitened means. to add mean back
% use ~ w*mean

% v_output = -1*w + [.7706; 1.4909];
% 
% (-1*w).*[.7706; 1.4909]
% 



%%%% ------ use the SVD to get the first principal component

[U S V] = svd(v_total);

u1 - [-0.323; -0.946];

figure; scatter(0.323, 0.946);     % gets principal component in between v1 and v2





% *** Next *** try equation 45,46 for 'several unit' ICA
% let W = W / sqrt( norm(WW') )
% 
% repeat: W = 3/2 * W  -  1/2 * WW'W
% 
% W is the matrix of w1, w2, wn vectors
% W = (w1, ... wn)'  w1 col vect is 1st row, w2 is 2nd row
%     take w1 ... wn to be the data points
%      since the fixed point iteration one-unit algorithm converges to the data point
%  
%  

x = v_zerod2;       % x variable as in UCLA ppr
 
% FAST ICA iteration --- for several units

W = rand(2,2)-0.5;          % 2 random vectors    -0.5 to 0.5 range


%%%% ---- iteration

diff_outer = 5;
counter_out = 0;

while (diff_outer > 1e-5)
    

% W_plus = W + Gamma( diag(-bi) + E(g(Wx)(Wx)')) *W
% 
% orthogonalize W with W=W/sqrt( norm(WW')),  W = 3/2W -1/2WW'W

% diag(-bi)
y = W*x;

                                    %tanh(2*w'*x2) - w*(1-(tanh(2*w'*x2))^2)
g_y = tanh(2*y);                   % 2x16 matrix of data points

gy_mult = y.*g_y;                  % y*g_y component wise

gy_expect = mean(gy_mult,2);       % mean of components (remove cols), get row vect


diag_term = -1*diag(gy_expect);    % -bi term

 
term2_temp = zeros(size(x,2),size(diag_term(:),1));    % 16 x 4 matrix
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
g_prime = 1-(tanh(2*y)).^2;              % 2x16

g_prime_expect = mean(g_prime,2);        % 2x1    'col vect that represents data'

term_g = gy_expect - g_prime_expect;
term_g = term_g.^(-1);               % -1 power   2x1 matrix
 
gamma = diag(term_g); 
 
 

% W_plus = W + Gamma*( diag(-bi) + E(g(Wx)(Wx)')) *W
% 
% orthogonalize W with W=W/sqrt( norm(WW')),  W = 3/2W -1/2WW'W

W_plus = W + gamma*(diag_term + term2)*W;      % update W

W_plus = W_plus/sqrt(norm(W_plus*W_plus',2));  % normalize


W_old = W_plus;
diff =5;

counter = 0;

while (diff > 1e-5)           % orthogonalize W

    W_plus = (3/2)*W_plus - 0.5*W_plus*W_plus'*W_plus;

    diff = norm(W_old-W_plus,2);

    W_old = W_plus;
    counter = counter+1;

end

diff_outer = norm(W-W_plus,2);

W = W_plus;          % update W to be orthogonalized W_plus;

counter_out = counter_out+1;

end  % outer iteration loop



%%%% --- converged after counter_out=10;

                % first W vect     -   points towards more of the points,
                % but between the v1 and v2 vects

figure; scatter(v_zerod2(1,:)', v_zerod2(2,:)'); title('whitened v_data'); 
        
hold on; plot([0 W(1,1)], [0 W(1,2)]); hold off;    % green point
hold on; plot([0 W(2,1)], [0 W(2,2)]); hold off;     % red point

% W  is orthogonal,    our vects are not




A = inv(W);         %%% --- look at A, the inverse of W

figure; scatter(v_zerod2(1,:)', v_zerod2(2,:)'); title('whitened v_data');
hold on; scatter(A(1,1), A(1,2)); hold off;    % green point ---W is vect w's transposed, take row to be component
hold on; scatter(A(2,1), A(2,2)); hold off;     % red point
hold on; plot([0 A(1,1)], [0 A(1,2)]); hold off;
hold on; plot([0 A(2,1)], [0 A(2,2)]); hold off;

% dewhiten data  (rotate 'projection' matrix)

W_dewhit = (U*(D.^(1/2))*U')*W;        % W_dewhit should be like W_i from FastICA
A_dewhit = (U*(D.^(1/2))*U')*A;


%%%%% ---------- U on W to rotate back into non whitened space
% W_nonw = U*W;  % need to rotate and stretch UDU'

%%%%% ---------- plot W-nonwhitened --------------
% figure; scatter(v_zerod(1,:)', v_zerod(2,:)'); title('orig data zero mean');
% hold on; scatter(W_nonw(1,1), W_nonw(2,1)); hold off;    % green point
% hold on; scatter(W_nonw(1,2), W_nonw(2,2)); hold off;     % red point
% hold on; plot([0 W_nonw(1,1)], [0 W_nonw(2,1)]); hold off;
% hold on; plot([0 W_nonw(1,2)], [0 W_nonw(2,2)]); hold off;

% A_nonw = inv(W_nonw); 
% %%%%% ----------- plot A non whitened --------------
% figure; scatter(v_zerod(1,:)', v_zerod(2,:)'); title('orig data zero mean');
% hold on; scatter(A_nonw(1,1), A_nonw(1,2)); hold off;    % green point
% hold on; scatter(A_nonw(2,1), A_nonw(2,2)); hold off;     % red point
% hold on; plot([0 A_nonw(1,1)], [0 A_nonw(1,2)]); hold off;
% hold on; plot([0 A_nonw(2,1)], [0 A_nonw(2,2)]); hold off;

%%%
%%%

% plot both orig vectors
figure; scatter(v_zerod(1,:)', v_zerod(2,:)'); title('orig data zero mean');
hold on; scatter(A(1,1), A(1,2)); hold off;    % green point
hold on; scatter(A(2,1), A(2,2)); hold off;     % red point


% plot dewhitened components
figure; scatter(v_total(1,:)', v_total(2,:)'); title('orig data W+ iteration');
hold on; plot([0 W_dewhit(1,1)], [0 W_dewhit(1,2)]); hold off;    % red point
hold on; plot([0 W_dewhit(2,1)], [0 W_dewhit(2,2)]); hold off;     % green point

                % A - mixing matrix
figure; scatter(v_zerod(1,:)', v_zerod(2,:)'); title('orig data');
hold on; scatter(A_dewhit(1,1), A_dewhit(1,2)); hold off;    % red point
hold on; scatter(A_dewhit(2,1), A_dewhit(2,2)); hold off;     % green point



% add mean back to vects
W_dewhit_mean(:,1) = W_dewhit(:,1) + W*v_mean;
W_dewhit_mean(:,2) = W_dewhit(:,2) + W*v_mean;



% plot dewhitened components
hold on; scatter(W_dewhit_mean(1,1), W_dewhit_mean(2,1)); hold off;    % red point

hold on; scatter(W_dewhit_mean(1,2), W_dewhit_mean(2,2)); hold off;     % green point

hold on; plot([0 W_dewhit_mean(1,1)], [0 W_dewhit_mean(2,1)]); hold off;
hold on; plot([0 W_dewhit_mean(1,2)], [0 W_dewhit_mean(2,2)]); hold off;




%%% --------- Fast ICA on vect v1,v2 whitened data

%[icasig, A_i, W_i] = fastica([example1; example2], 'numOfIC', 2);
                           % examples in rows
[icasig, A_i, W_i] = fastica(v_total, 'numOfIC', 2, 'maxNumIterations', 1000000);
                            %v_total or v_zerod2
% plot both vectors
figure; scatter([v2_total(:,1); v1_total(:,1)], [v2_total(:,2); v1_total(:,2)]);
title('original data FASTICA');

% plot both vectors - whitened
figure; scatter(v_zerod2(1,:)', v_zerod2(2,:)'); title('whitened v_data');

                           % rows of A give directions
hold on; plot([0 A_i(1,1)], [0 A_i(1,2)]); hold off;   % get princip component direction
hold on; plot([0 A_i(2,1)], [0 A_i(2,2)]); hold off;   % get princip component direction



                           % rows of W give directions
hold on; plot([0 W_i(1,1)], [0 W_i(1,2)]); hold off;   % get princip component direction
hold on; plot([0 W_i(2,1)], [0 W_i(2,2)]); hold off;   % get princip component direction


%%%% ---- try U*d^-1/2 *U' to get to whitened space

W_i_rot = (U*d2*U')*W_i;


% plot both vectors - whitened
figure; scatter(v_zerod2(1,:)', v_zerod2(2,:)'); title('whitened v_data');
hold on; plot([0 W_i_rot(1,1)], [0 W_i_rot(1,2)]); hold off;   % get princip component direction
hold on; plot([0 W_i_rot(2,1)], [0 W_i_rot(2,2)]); hold off;   % get princip component direction





