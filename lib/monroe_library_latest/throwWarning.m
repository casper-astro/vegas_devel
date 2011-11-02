function throwWarning(errorMsg)

%strError = strcat(errorMsg, char(10), char(13),  ';  <in: ', pathToBlock, ' >');
strError = strcat(errorMsg, char(10), char(13));

%disp(strError);
warning(strError);