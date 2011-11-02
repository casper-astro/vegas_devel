function i = getVerticalPlaceLoc(placementArray, col)

i=1;

while(~strcmp(placementArray(col,i).instName, 'none'))
   i=i+1; 
end

