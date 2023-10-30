function [Rnew, Rnewindx, max_Rindx,policy,Valutogo_update]=Planning_Bayesian(cycle,n,m,Pest,H,Q,F,ExpSurprise_previous,R_Localized,R_Localized_indx,Valuetogo,alpha,gamma,epsilon,policy,TERM1_ExpSurp_prev,TERM2_ExpSurp_prev)
R_Loc=size(R_Localized,3);
Spred_new=zeros(m,m,R_Loc); 

reward_Bayesian=zeros(1,R_Loc);
TERM1_ExpSurp=zeros(1,R_Loc);
TERM2_ExpSurp=zeros(1,R_Loc);
ExpSurprise=zeros(1,R_Loc);
Valutogo_update=zeros(1,R_Loc);
Valpol=zeros(1,R_Loc);

Ppred = F*Pest*F' + Q;

if cycle==2
 for i=1:R_Loc
  Spred_new(:,:,i)=R_Localized(:,:,i) + H*Ppred*H';
  ExpSurprise(i)= trace(pinv(R_Localized(:,:,i)*pinv(Spred_new(:,:,i))))+ 0.5*log(det(R_Localized(:,:,i)*pinv(Spred_new(:,:,i))))-m;  
  
  TERM1_ExpSurp(i)=det(R_Localized(:,:,i)*pinv(Spred_new(:,:,i)));
  TERM2_ExpSurp(i)=det(eye(m)-(R_Localized(:,:,i)*pinv(Spred_new(:,:,i))));

  reward_Bayesian(i)= ((TERM1_ExpSurp_prev + (ExpSurprise_previous*TERM2_ExpSurp_prev))/(1+ExpSurprise_previous)) -((TERM1_ExpSurp(i) + (ExpSurprise(i)*TERM2_ExpSurp(i)))/(1+ExpSurprise(i)));
  
  Valutogo_update(i)=Valuetogo(1,R_Localized_indx(i))+ alpha*(reward_Bayesian(i)+ gamma*( mean(Valuetogo(1,R_Localized_indx)) ) - Valuetogo(R_Localized_indx(i)) );
 end
else
 for i=1:R_Loc
  Spred_new(:,:,i)=R_Localized(:,:,i) + H*Ppred*H';
  ExpSurprise(i)= trace(pinv(R_Localized(:,:,i)*pinv(Spred_new(:,:,i))))+ 0.5*log(det(R_Localized(:,:,i)*pinv(Spred_new(:,:,i))))-m;  
  
   TERM1_ExpSurp(i)=det(R_Localized(:,:,i)*pinv(Spred_new(:,:,i)));
   TERM2_ExpSurp(i)=det(eye(m)-(R_Localized(:,:,i)*pinv(Spred_new(:,:,i))));
 
   reward_Bayesian(i)= ((TERM1_ExpSurp_prev + (ExpSurprise_previous*TERM2_ExpSurp_prev))/(1+ExpSurprise_previous)) -((TERM1_ExpSurp(i) + (ExpSurprise(i)*TERM2_ExpSurp(i)))/(1+ExpSurprise(i)));
    
   RR_Localized=R_Localized;
   RR_Localized_indx=R_Localized_indx;
   Filename=sprintf('%d.mat', R_Localized_indx(1,i));
   load(['C:\Users\Yeganeh\Documents\Kalman Filter Examples\January 2021\Autonomous Car\Radar\Making_Rmatrix\Rmatrix' Filename],'R_Localized','R_Localized_indx');
   Valpol(i)=Valuetogo(1,R_Localized_indx)*(policy(RR_Localized_indx(i),R_Localized_indx))';  
   Valutogo_update(i)=Valuetogo(RR_Localized_indx(i))+ alpha*(reward_Bayesian(i)+ gamma*(Valpol(i)) - Valuetogo(RR_Localized_indx(i)) );  
   R_Localized=RR_Localized;
   R_Localized_indx=RR_Localized_indx; 
  
 end
end 

eta=rand;
policy(R_Localized_indx(1),R_Localized_indx)=(epsilon/R_Loc)*ones(1,R_Loc);
if eta>= epsilon 
 max_Rindx=find(Valutogo_update == max(Valutogo_update(:)));   
 Rnewindx=max_Rindx; 
 policy(R_Localized_indx(1,1),R_Localized_indx(1,Rnewindx))=1-epsilon+epsilon/R_Loc;
 Rnew=R_Localized(:,:,Rnewindx);
 mark(1,cycle)=0;
else
max_Rindx=find(Valutogo_update == max(Valutogo_update(:)));
Rnewindx=randperm(R_Loc,1);  
policy(R_Localized_indx(1,1),R_Localized_indx(1,max_Rindx))=1-epsilon+epsilon/R_Loc;    
Rnew=R_Localized(:,:,Rnewindx); 
mark(1,cycle)=1;
end




