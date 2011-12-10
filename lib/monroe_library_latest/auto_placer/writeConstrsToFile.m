function writeConstrsToFile(constrs, fileName, append)

if(~append)
    id = fopen(fileName, 'w+');
else
    id = fopen(fileName, 'a+');
end

for(i=1:length(constrs))
    fprintf(id, '%s', constrs{i});
    fwrite(id, char(10));
end

fclose(id);

