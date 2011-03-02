t = 0:2^16;
t = t'/8;
%din = .9*sin(2*pi*t/8.);
din = chirp(t,1/16.0,t(end),4.0);
vals=[t,din];
