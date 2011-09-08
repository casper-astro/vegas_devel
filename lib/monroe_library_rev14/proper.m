function res = proper(strIn)

strIn(1) = upper(strIn(1));
strIn(2:length(strIn)) = lower(strIn(2:length(strIn)));

res = strIn;