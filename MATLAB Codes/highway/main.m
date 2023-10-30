clc
clear all

% This program implements the system performance of our proposed learning
% algorithm based on the Expectation of Bayesian Surprise for the highway 
% driving scenario with constant acceleration 
% run the Making_xMC file first and than run the main file


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
z=zeros(m,K);     % measurement vector

xpred=zeros(n,K); % predicted state vector k|k-1
xest=zeros(n,K);  % estimate state vector k|k


% Error Covariance
Ppred=zeros(n,n,K);   % predicted error covariance P(k|k-1)
Pest=zeros(n,n,K);    % estimate error covariance at P(k|k) 
Spred=zeros(m,m,K);   % predicted innovation covariance matrix 
Sest=zeros(m,m,K);    % estimated innovation covariance matrix
nu=zeros(m,K);        % innovation vector
KG=zeros(n,m,K);      % Kalman Gain matrix

NORM2est_ind_MC=zeros(n,K,MC);  % records the norm 2 vector between xest and x for all Monte Carlo simulations 
ExpSurprise_MC=zeros(MC,K);     % computes the expectation of Bayesian surprise for all Monte Carlo simulations
RMSE_ind=zeros(n,K);            % Root Mean Square Error (RMSE)  

% Setting the parameters of the simulation by initializing the state-space model: 

x0 =[25; 3; 100; 23; 2];      % true initial state vector
xpred0=[24; 3; 80; 23; 2];    % initial state estimation mean vector 
Ppred0=diag([100,1,100, 100, 1]);    % initial state estimation covariance matrix
distance=100;    % initial distance between two vehicles
B=10^(8);        % Bandwidth 

landa=[10^(-6):0.1*10^(-6):10^(-5), 1.1*10^(-5):0.1*10^(-5):10^(-4)] ; % pulse duration values
beta= [-10^(12) :0.2*10^(12): -0.2*10^(12),   0.2*10^(12):0.2*10^(12):10^(12)]; % chirp rate values
R_library=Select_Measurementnoise(landa,beta,distance,B);       % This function builds the whole action library (measurement 
                                                                % noise covariance library)
NR = 1: (length(landa)*length(beta));      % counts the action library 

% setting the values of the reinforcement learning algorthim
alpha=0.1;    % learning rate
gamma=0.5;    % discount factor
epsilon=0.1;  % e-greedy algorithm parameter exploration vs explotation factor

SEED=200;     % setting the seed of the noise 
rng(SEED);

for mc=1:MC    

% Select measurement noise R
NRR = NR(randperm(length(NR),1));  % Select the initial action index randomly based on uniform distirbution 
R=R_library(:,:,NRR);              % set the action

% Set initial conditions of the filter
x(:,1)= x0;
xpred(:,1)= xpred0; 
xest(:,1)= xpred0;
Ppred(:,:,1) =Ppred0;
Pest(:,:,1) =Ppred0;
Spred(:,:,1)=R + H*Pest(:,:,1)*H'; 
KG(:,:,1)=Pest(:,:,1)*H'*pinv(R);
z(:,1)=H*x(:,1);
nu(:,1)=z(:,1)-H*xpred(:,1);


% Define the vectors for the Expectation of Bayesian Surprise 

TERM1_ExpSurp=zeros(1,K); 
TERM2_ExpSurp=zeros(1,K);
TERM3_ExpSurp=zeros(1,K);

% initialize the learning and planning elements of the algorithm: the
% reward function, Value-to-go-function and the policy function 
reward=ones(1,K); % reward function
Valuetogo=ones(1,(length(landa)*length(beta))); % Initialize value to go function
policy=ones((length(landa)*length(beta)),(length(landa)*length(beta)));  % policy 

Rindx=zeros(1,K);     % record the index of the selected actions at each time instant 
Rindx(1,1)=NRR;       % the index of the intial action 
max_Rindx=zeros(1,K); % the index of the action with maximum value-to-go-function
max_Rindx(1,1)=NRR;   %  set the index same as the initial action

% Compute the Bayesian surprise and its expectation at time k=0 (initial time)
Bayesian_surprise(1,1)= 0.5 *( nu(:,1)'*KG(:,:,1)'*pinv(Pest(:,:,1))*KG(:,:,1)*nu(:,1) + trace(Spred(:,:,1)*pinv(R))-m+ log(det(R*pinv(Spred(:,:,1))))  );
ExpSurprise(1,1)= trace(pinv(R*pinv(Spred(:,:,1))))+ 0.5*log(det(R*pinv(Spred(:,:,1))))-m;  

% Intialize the proposed reward based on the expectation of Bayesian surprise inspired by BERF 
TERM1_ExpSurp(1,1)=det(R*pinv(Spred(:,:,1)));
TERM2_ExpSurp(1,1)=det(eye(m)-(R*pinv(Spred(:,:,1))));
TERM3_ExpSurp(1,1)=det(pinv(R*pinv(Spred(:,:,1)))-eye(m));
reward(1,1)= (TERM1_ExpSurp(1,1)+(ExpSurprise(1,1)*TERM2_ExpSurp(1,1)))/(1+ExpSurprise(1,1));

R_Loc=25; % Size of the localized action library (number of actions in the action library)

% This part loads the MATLAB file for the state vector x_k that we have 
% implemented randomly for constant acceleration (highway driving experience)   
load('testMC.mat','x_MC');
x=x_MC(:,:,mc);


for j=2:K
  
 z(:,j)=H*x(:,j)+ mvnrnd(zeros(1,m),R)';  % Received measurement vector 

 
 %% Kalman Filter Estimation and Prediction

  [xpred(:,j),Ppred(:,:,j),Spred(:,:,j)]=timeupdate(xest(:,j-1),Pest(:,:,j-1),R,H,Q,F); 
  [xest(:,j), Pest(:,:,j), KG(:,:,j),nu(:,j)] = measurementupdate(xpred(:,j),Ppred(:,:,j),Spred(:,:,j),z(:,j),H);  


  % creates a localized action library using k-nearest neighbor algorithm
  [R_Localized,  R_Localized_indx]=nearest_neighbour(landa,beta, Rindx(1,j-1),R_library,R_Loc); 
   
 %% Information Processor: Computes the Bayesian Surprise and its expectation

  Bayesian_surprise(1,j)= 0.5 *( nu(:,j)'*KG(:,:,j)'*pinv(Pest(:,:,j))*KG(:,:,j)*nu(:,j) + trace(Spred(:,:,j)*pinv(R))-m+ log(det(R*pinv(Spred(:,:,j))))  );
  ExpSurprise(1,j)= trace(pinv(R*pinv(Spred(:,:,j))))+ 0.5*log(det(R*pinv(Spred(:,:,j))))-m;  
  
 %% Computes the proposed Reward Function based on expectation of Bayesian surprise
  TERM1_ExpSurp(1,j)=det(R*pinv(Spred(:,:,j)));
  TERM2_ExpSurp(1,j)=det(eye(m)-(R*pinv(Spred(:,:,j))));
  reward(1,j)= (TERM1_ExpSurp(1,j-1) + (ExpSurprise(1,j-1)*TERM2_ExpSurp(1,j-1)))/(1+ExpSurprise(1,j-1))  -((TERM1_ExpSurp(1,j) + (ExpSurprise(1,j)*TERM2_ExpSurp(1,j)))/(1+ExpSurprise(1,j)));
   
 %% Learning Stage: Updating Value-to-go function 
   if j==2
      Valuetogo(1,Rindx(1,j-1))=Valuetogo(1,Rindx(1,j-1)) + alpha*(reward(1,j)+ gamma*(mean(Valuetogo(1,R_Localized_indx)))-Valuetogo(1,Rindx(1,j-1)))  ;
   else     
      Valpol=Valuetogo(1,R_Localized_indx)*(policy(Rindx(1,j-1),R_Localized_indx))';
      Valuetogo(1,Rindx(1,j-1))=Valuetogo(1,Rindx(1,j-1)) + alpha*(reward(1,j)+ gamma*(Valpol) - Valuetogo(1,Rindx(1,j-1)))  ; 
   end
      
  %% Planning and Policy Stage
   [Rnew, Rnewindx, maxRindx,policy,Valutogo_update]=Planning_Bayesian(j,n,m,Pest(:,:,j),H,Q,F,ExpSurprise(1,j),R_Localized,R_Localized_indx,Valuetogo,alpha,gamma,epsilon,policy,TERM1_ExpSurp(1,j),TERM2_ExpSurp(1,j));

  %% Action Selection 
   R=Rnew;      % Apply the new action 
   Rindx(1,j)= R_Localized_indx(1,Rnewindx); % record the new action
   max_Rindx(1,j)=R_Localized_indx(1,maxRindx); 

 end
 %% Computing the second norm to calculate RMSE  

 [NORM1est_ind,NORM1_ind,NORM2est_ind,NORM2_ind] = Performance_Test(K,n,x, xest); 
 NORM2est_ind_MC(:,:,mc)=NORM2est_ind;
 ExpSurprise_MC(mc,:)=ExpSurprise; % recording the Expectation of Bayesian surprise

mc % print the Monte Carlo simulation run
end

% compute the average of the expectation of Bayesian surprise over all Monte Carlo runs
AVERG_ExpSurprise=1/MC*sum(ExpSurprise_MC,1);

% compute the average of RMSE over all Monte Carlo runs 
NOM1=sum(NORM2est_ind_MC,3); 
for j=1:K
    for i=1:n
    RMSE_ind(i,j)=sqrt((1/MC).*NOM1(i,j));
    end
end

%% Plotting system performance 

% plot the RMSE performance of our proposed approach for estimating the longtitude distance
f=2:K;  
figure, 
p= plot(f, (log10(RMSE_ind(3,f))), 'b -');
p(1).LineWidth = 3;
leg2 = legend('\textbf{RMSE}');
set(leg2,'Interpreter','latex');
set(leg2,'FontSize',22);
xlabel({'\textbf{Time Samples($k$)}'},'Interpreter','latex','FontSize',22,'Color','k'); 
ylabel('$\mathbf{E}[\mathcal{S}^{B}_{k}] (d_x)$','Interpreter','latex','FontSize',22,'Color','k');
set(gca,'FontSize',22)
set(gca,'FontName','Times New Roman')
set(gcf,'color','white')
hold on  

% plot the RMSE performance of our proposed approach for estimating the target velocity
figure, 
p= plot(f, (log10(RMSE_ind(4,f))), 'b -');
p(1).LineWidth = 3;
leg2 = legend('\textbf{RMSE}');
set(leg2,'Interpreter','latex');
set(leg2,'FontSize',22);
xlabel({'\textbf{Time Samples($k$)}'},'Interpreter','latex','FontSize',22,'Color','k'); 
ylabel('$\mathbf{E}[\mathcal{S}^{B}_{k}] (v^1_x)$','Interpreter','latex','FontSize',22,'Color','k');
set(gca,'FontSize',22)
set(gca,'FontName','Times New Roman')
set(gcf,'color','white')
hold on  

% plot the Average of the Expectation of Bayesian surprise over all Monte
% Carlo runs
figure,
p=plot(f, log10(AVERG_ExpSurprise(1,f)),'-r');
p(1).LineWidth = 3;
leg2 = legend('\textbf{Expectation of Bayesian Surprise}');
set(leg2,'Interpreter','latex');
set(leg2,'FontSize',22);
xlabel('\textbf{Time Sample $k$}','Interpreter','latex','FontSize',22,'Color','k');
ylabel('\textbf{Expectation of Bayesian Surprise}','Interpreter','latex','FontSize',22,'Color','k');
set(gca,'FontSize',22)
set(gca,'FontName','Times New Roman')
set(gcf,'color','white')
hold on  
