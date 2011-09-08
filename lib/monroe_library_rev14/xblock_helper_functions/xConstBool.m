function sOut = xConstBool(value, name)

if((value ~= 0) && (value ~= 1))
   throwError(strcat('value must be either 0 or 1; value = ' , num2str(value)));
end


sOut = xSignal;

bConstantBool = xBlock(struct('source', 'Constant', 'name', name), ...
    struct('arith_type', 'Boolean', 'const', value, 'explicit_period', 'on'), ...
    {}, {sOut});
