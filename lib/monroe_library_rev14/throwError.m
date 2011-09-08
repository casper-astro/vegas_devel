function throwError(errorMsg)

%strError = strcat('ERROR: ', errorMsg, char(10), char(13), ';  <error in: ', pathToBlock, ' >');
strError = strcat('ERROR: ', errorMsg, char(10), char(13));

disp(strError);
error(strError);

