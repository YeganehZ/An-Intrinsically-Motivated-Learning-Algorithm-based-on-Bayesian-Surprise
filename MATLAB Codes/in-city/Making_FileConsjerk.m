clc
clear all

K=70;  % time index 
MC=100;  % number of Monte Carlo runs

delta=0.1; % sample time
q=0.01;    % state noise variance 

% Transition Matrix
F = [1 delta delta^2/2 0 0 0 0; ...
     0 1 delta 0 0 0 0;...
     0 0 1 0 0 0 0;...
     -delta -delta^2/2 -delta^3/6 1 delta delta^2/2 delta^3/6;...
     0 0 0 0 1 delta delta^2/2;...
     0 0 0 0 0 1 delta;...
     0 0 0 0 0 0 1]; 

% Measurement Matrix 
H=[0 0 0 1 0 0 0;0 0 0 0 1 0 0]; 

% State Noise Covariance
Q = [(delta^6)/36  (delta^5)/12    (delta^4)/6   -(delta^7)/144  0   0  0; ...
    (delta^5)/12     (delta^4)/4   (delta^3)/2   -(delta^6)/48   0   0  0; ...
    (delta^4)/6     (delta^3)/2    (delta^2)     -(delta^5)/24   0   0  0; ...    
    -(delta^7)/144  -(delta^6)/48   -(delta^5)/24  (delta^8)/288   (delta^7)/144  (delta^6)/48   (delta^5)/24;...
        0            0                0               (delta^7)/144   (delta^6)/36   (delta^5)/12   (delta^4)/6;...    
        0            0                0               (delta^6)/48    (delta^5)/12   (delta^4)/4   (delta^3)/2;...
        0            0                0               (delta^5)/24    (delta^4)/6    (delta^3)/2     delta^2]*q; 

n=size(H,2);   % Dimension of state vector, state estimate, .... 
m=size(H,1);   % Dimension of the measurement vector, innovation sequence

% Pre-setting the vectors and matrices:
x=zeros(n,K);     % state vector

% Setting the parameters of the simulation by initializing the state-space model for in-city driving: 
x0 =[16.7; 0;0; 27.8; 13.9; 0;0];       % true initial state vector
x_MC=zeros(n,K,MC);


%% Creating the testMC_Consjerk.mat 

for mc=1:MC    
% Initial Conditions
x(:,1)= x0;
f1=10;
  for j=1:f1
    x(:,j+1)=F*x(:,j)+ mvnrnd(zeros(1,n),Q)';  
  end
x(:,f1+1) =[16.7; -0.12;-1.2; 25; 13.9; -0.15; -1.5];
f2=52;
  for j=f1+1:f2
    x(:,j+1)=F*x(:,j)+ mvnrnd(zeros(1,n),Q)';  
  end 
fdis=x(4,f2+1);
facc=x(2,f2+1);
fvel=x(1,f2+1);
fjer=x(3,f2+1);
x(:,f2+1) =[fvel; facc; fjer; fdis; 0; 0;0];
% x(:,f2+1) =[8.85; -2.02; -0.78; 10.78; 0; 0;0];
f3=62;
 for j=f2+1:f3
   x(:,j+1)=F*x(:,j)+ mvnrnd(zeros(1,n),Q)';  
 end   
fdis2=x(4,f3+1); 
x(:,f3+1) =[0; 0; 0; fdis2; 0; 0;0];
 for j=f3+1:K-1
   x(:,j+1)=F*x(:,j)+ mvnrnd(zeros(1,n),Q)';  
 end 
x_MC(:,:,mc)=x; 
mc  
end

save('testMC_Consjerk.mat','x_MC')
load('testMC_Consjerk.mat','x_MC')


%% Plotting One Monte Carlo run for the state-values:
x=x_MC(:,:,50);
f=1:K;

figure, 
p= plot(f, x(1,f),'b -',f, x(5,f),'b --');
p(1).LineWidth = 3;
p(2).LineWidth = 3;
leg2 = legend('\textbf{$v^0_x$}', '$v^1_x$');
set(leg2,'Interpreter','latex');
set(leg2,'FontSize',22);
xlabel({'\textbf{Time Samples $k$ }'},'Interpreter','latex','FontSize',22,'Color','k'); 
ylabel('\textbf{$v^0_x$ and $v^1_x$}','Interpreter','latex','FontSize',22,'Color','k');
set(gca,'FontSize',22)
set(gca,'FontName','Times New Roman')
set(gcf,'color','white')
hold on   

figure, 
p= plot(f, x(2,f),'r -',f, x(6,f),'r -');
p(1).LineWidth = 3;
p(2).LineWidth = 3;
leg2 = legend('$a^0_x$', '$a^1_x$');
set(leg2,'Interpreter','latex');
set(leg2,'FontSize',22);
xlabel({'\textbf{Time Samples $k$ }'},'Interpreter','latex','FontSize',22,'Color','k'); 
ylabel('\textbf{$a^0_x$ and $a^1_x$}','Interpreter','latex','FontSize',22,'Color','k');
set(gca,'FontSize',22)
set(gca,'FontName','Times New Roman')
set(gcf,'color','white')
hold on   

figure, 
p= plot(f, x(3,f),'k -',f, x(7,f),'k --');
p(1).LineWidth = 3;
p(2).LineWidth = 3;
leg2 = legend('$j^0_x$', '$j^1_x$');
set(leg2,'Interpreter','latex');
set(leg2,'FontSize',22);
xlabel({'\textbf{Time Samples $k$ }'},'Interpreter','latex','FontSize',22,'Color','k'); 
ylabel('\textbf{$j^0_x$ and $j^1_x$}','Interpreter','latex','FontSize',22,'Color','k');
set(gca,'FontSize',22)
set(gca,'FontName','Times New Roman')
set(gcf,'color','white')
hold on   

figure, 
p= plot(f, x(4,f),'c -');
p(1).LineWidth = 3;
leg2 = legend('$d_x$');
set(leg2,'Interpreter','latex');
set(leg2,'FontSize',22);
xlabel({'\textbf{Time Samples $k$ }'},'Interpreter','latex','FontSize',22,'Color','k'); 
ylabel('\textbf{$d_x$}','Interpreter','latex','FontSize',22,'Color','k');
set(gca,'FontSize',22)
set(gca,'FontName','Times New Roman')
set(gcf,'color','white')
hold on   



