clc
clear all

K=100;  % time index 
MC=100;  % number of Monte Carlo runs

delta=0.1; % sample time
q=0.01;    % state noise variance

% Transition Matrix
F = [1 delta 0 0 0; ...
    0 1 0 0 0;...
    -delta -delta^2/2 1 delta delta^2/2;...
    0 0 0 1 delta;...
    0 0 0 0 1]; 

% Measurement Matrix
H=[0 0 1 0 0; 0 0 0 1 0]; 

% State Noise Covariance
Q = [(delta^4)/4    (delta^3)/2   -(delta^5)/12  0   0; ...
     (delta^3)/2     delta^2     -(delta^4)/6   0   0; ...
    -(delta^5)/12  -(delta^4)/6   (delta^6)/18  (delta^5)/12   (delta^4)/6;...
        0            0              (delta^5)/12   (delta^4)/4   (delta^3)/2;...
        0            0              (delta^4)/6   (delta^3)/2     delta^2]*q; 

    
n=size(H,2);   % Dimension of state vector, state estimate, .... 
m=size(H,1);   % Dimension of the measurement vector, innovation sequence

% Pre-setting the vectors and matrices:

x=zeros(n,K);     % state vector



x0 =[25; 3; 100; 23; 2];
x_MC=zeros(n,K,MC);    
distance=100;    
B=10^(8);

landa=[10^(-6):0.1*10^(-6):10^(-5), 1.1*10^(-5):0.1*10^(-5):10^(-4)] ; 
beta= [-10^(12) :0.2*10^(12): -0.2*10^(12),   0.2*10^(12):0.2*10^(12):10^(12)];  

R_library=Select_Measurementnoise(landa,beta,distance,B);
NR = 1: (length(landa)*length(beta));


SEED=200;
rng(SEED);

for k=1:MC    
  for j=1:K-1
    x(:,j+1)=F*x(:,j)+ mvnrnd(zeros(1,n),Q)';  
  end
  x_MC(:,:,k)=x;
  k
end

save('testMC.mat','x_MC')
