function hold_fir_init_xblock(blk, varargin)


defaults = {'add_latency', 2, ...
    'mult_latency', 3,...
    'hold_period', 1, ...
    'coeffs', [1.0, 1.0, 1.0]};


add_latency = get_var('add_latency', 'defaults', defaults, varargin{:});
mult_latency = get_var('mult_latency', 'defaults', defaults, varargin{:});
hold_period = get_var('hold_period', 'defaults', defaults, varargin{:});
coeffs = get_var('coeffs', 'defaults', defaults, varargin{:});

%% inports
in = xInport('in');
sync = xInport('sync');

%% outports
out = xOutport('out');
en_out = xOutport('en_out');

%% diagram
nonzero_ind = find(coeffs);
len = length(nonzero_ind);


% add enable signal generator
en = xSignal('en');
hold_fir_en = xBlock(struct('source', str2func('hold_fir_en_init_xblock'), 'name', 'en_gen'), ...
    {[blk, '/en_gen'], ...
    'hold_period', hold_period}, ...
    {sync}, ...
    {en});

% add delay blocks
Delay_ins = cell(len, 1);
Delay_blks = cell(len-1, 1);
Delay_ins{1} = in;
for i =1:length(coeffs)-1
    % block: myfir_hold_test/myFIR/Delay
    Delay_ins{i+1} = xSignal(['xlsub2_Delay_out', num2str(i)]);
    Delay_blks{i} = xBlock(struct('source', 'Delay', 'name', ['Delay',num2str(i)]), ...
                          struct('latency', hold_period), ...
                          Delay_ins(i), ...
                          Delay_ins(i+1));
end

% add taps
myfir_tap = cell(len,1);
myfir_tap_out = cell(len,1);
for i = 1:len
    myfir_tap_out{i} = xSignal(['hold_fir_tap_out',num2str(i)]);
    myfir_tap{i} = xBlock(struct('source',  str2func('hold_fir_tap_init_xblock'), 'name', ['hold_fir_tap',num2str(nonzero_ind(i))]), ...
                              {[blk, '/hold_fir_tap',num2str(nonzero_ind(i))], ...
                                'mult_latency', mult_latency, ...
                                'hold_period', hold_period, ...
                                'coefficient', coeffs(nonzero_ind(i))}, ...
                              {Delay_ins{nonzero_ind(i)}, en}, ...
                              {myfir_tap_out{i}, []});
end



% add adder tree
adder_tree = xBlock(struct('source', str2func('hold_fir_adder_tree_init_xblock'), 'name', 'adder_tree'), ...
    {[blk, '/adder_tree'], ...
    'add_latency', add_latency, ...
    'hold_period', hold_period, ...
    'n_inputs', len}, ...
    {en, myfir_tap_out{:}}, ...
    {en_out, out});



if ~isempty(blk) && ~strcmp(blk(1),'/')
    clean_blocks(blk);
    fmtstr=sprintf('%d-tap FIR filter (%d nonzero tap)\n hold period: %d\n mult latency: %d\n add latency: %d',length(coeffs),len, hold_period, mult_latency,add_latency);
    set_param(blk,'AttributesFormatString',fmtstr);
end


end

