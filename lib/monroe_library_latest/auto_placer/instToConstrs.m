function constrs = instToConstrs(inst, type, row, col)

emptyPlacementSlot.instName = 'none';
emptyPlacementSlot.rlocGroup = 'none';
emptyPlacementSlot.rloc = 'none';
emptyPlacementSlot.rlocRange = 'none';
emptyPlacementSlot.rlocInst{1} = 'none';

% if( strcmp(inst.instName, 'reserved_bram36'))
%     
%     2-2
% end


if(strcmp(inst.instName, 'none') || strcmp(inst.instName, 'reserved_bram36') || strcmp(inst.instName, 'reserved') || strcmp(inst.instName, 'reserved_bram36 (suprise!)'))
    constrs = {''};
    return;
end


if(strcmp(type, 'dsp'))
    [grid_x, grid_y]=dspPlaceToRpmGrid(col, row);
    typeStr = 'DSP48_';
    row = row-1;
    
else
    [grid_x, grid_y]=bramPlaceToRpmGrid(col, row);
    typeStr = 'RAMB36_';
    row = floor((row-1) /2);
end

%LOC
constrs{1} = strcat(inst.instName, ' RPM_GRID = GRID;');
constrs{2} = strcat(inst.instName, ' LOC = ', typeStr , 'X', num2str(col-1), 'Y', num2str(row-1), ';');

%rloc (if any)
if(~strcmp(inst.rlocRange, 'none'))
   % constrs{3} = strcat(inst.instName, ' RLOC = ', inst.rloc, ';');
    constrs{3} = strcat(inst.instName, ' U_SET = ', inst.rlocGroup, ';');
    constrs{4} = strcat(inst.instName, ' RLOC_RANGE = ', inst.rlocRange, ';');
    
    i=1;
    %while(~strcmp(inst.rlocInst(i).instName, 'none'))
    if(~strcmp(inst.rlocInst{1}, 'none'))
        for(i=1:length(inst.rlocInst))
            index= i*2+3;
            constrs{index} = strcat(inst.rlocInst{i}, ' RPM_GRID = GRID;');
            constrs{index+1} = strcat(inst.rlocInst{i}, ' U_SET = ', inst.rlocGroup, ';');
            %i=i+1;
        end
    end
end