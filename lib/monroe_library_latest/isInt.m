function res = isInt(val)
if(islogical(val))
    res = 1;
else
    res = (floor(val) == val);
    res = min(res);
end