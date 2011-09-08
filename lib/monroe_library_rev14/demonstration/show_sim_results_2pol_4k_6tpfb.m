index1=11449+1025-47; vector_len=4096/4;
close all
index2=index1+vector_len-1;

temp2_0= [simout0.signals.values(index1:index2)' simout1.signals.values(index1:index2)' simout2.signals.values(index1:index2)' simout3.signals.values(index1:index2)']';
temp2_1 =[simout4.signals.values(index1:index2)' simout5.signals.values(index1:index2)' simout6.signals.values(index1:index2)' simout7.signals.values(index1:index2)']';

index1=index1-3; vector_len=4096/4;

index2=index1+vector_len-1;

temp3_0= [simout8.signals.values(index1:index2)' simout9.signals.values(index1:index2)' simout10.signals.values(index1:index2)' simout11.signals.values(index1:index2)']';
temp3_1= [simout12.signals.values(index1:index2)' simout13.signals.values(index1:index2)' simout14.signals.values(index1:index2)' simout15.signals.values(index1:index2)']';

%temp4= [simout_full0.signals.values(index1:index2)' simout_full1.signals.values(index1:index2)' simout_full2.signals.values(index1:index2)' simout_full3.signals.values(index1:index2)' simout_full4.signals.values(index1:index2)' simout_full5.signals.values(index1:index2)' simout_full6.signals.values(index1:index2)' simout_full7.signals.values(index1:index2)' ]';

% for i=1:61
%    index1=index1+vector_len; index2=index1+vector_len-1;
%    
%    temp3= temp3 + [simout8.signals.values(index1:index2)' simout9.signals.values(index1:index2)' simout10.signals.values(index1:index2)' simout11.signals.values(index1:index2)' simout12.signals.values(index1:index2)' simout13.signals.values(index1:index2)' simout14.signals.values(index1:index2)' simout15.signals.values(index1:index2)' ]';
% 
%  %  temp4= temp4 + [simout_full0.signals.values(index1:index2)' simout_full1.signals.values(index1:index2)' simout_full2.signals.values(index1:index2)' simout_full3.signals.values(index1:index2)' simout_full4.signals.values(index1:index2)' simout_full5.signals.values(index1:index2)' simout_full6.signals.values(index1:index2)' simout_full7.signals.values(index1:index2)' ]';
% 
% end


%figure();plot(0:vector_len*8-1,temp2); title('accumulated,linear'); 
figure();semilogy(0:vector_len*4-1,temp2_0); title('accumulated,pol0, log-y');
figure();semilogy(0:vector_len*4-1,temp2_1); title('accumulated,pol1, log-y');

%figure();plot(0:vector_len*8-1,temp3); title('raw,linear'); 
figure();semilogy(0:vector_len*4-1,temp3_0); title('raw,pol0, log-y');
figure();semilogy(0:vector_len*4-1,temp3_1); title('raw,pol1, log-y');

%figure();plot(0:vector_len*8-1,temp4); title('raw : wider,linear'); figure();semilogy(0:vector_len*8-1,temp4); title('raw : wider,log-y');
