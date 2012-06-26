function mux_fir_init_xblock(blk, varargin)


defaults = {'add_latency', 2, ...
    'mult_latency', 3,...
    'mux_latency', 1, ...
    'hold_period', 2, ...
    'coeffs', [0.1, 0.0, 0.2, 0.3, 0.0, 0.4, 0.5], ...
    'n_bits', 16, ...
    'bin_pt', 14};


add_latency = get_var('add_latency', 'defaults', defaults, varargin{:});
mult_latency = get_var('mult_latency', 'defaults', defaults, varargin{:});
mux_latency = get_var('mux_latency', 'defaults', defaults, varargin{:});
hold_period = get_var('hold_period', 'defaults', defaults, varargin{:});
n_bits = get_var('n_bits', 'defaults', defaults, varargin{:});
bin_pt = get_var('bin_pt', 'defaults', defaults, varargin{:});
coeffs = get_var('coeffs', 'defaults', defaults, varargin{:});


coeffs = coeffs(end:-1:1);
if any(gt(coeffs, 2^(n_bits-bin_pt-1)))
    disp('overflow occur! coefficients too big!');
    coeff_msg = ' overflow!';
else
    coeff_msg = '';
end



%% inports
in = xInport('in');
sync = xInport('sync');

%% outports
out = xOutport('out');
en_out = xOutport('en_out');

%% diagram
nonzero_ind = find(coeffs);
len = length(nonzero_ind);
len_total = length(coeffs);


in_delayed = xSignal('in_delayed');
in_delay = xBlock(struct('source', 'Delay', 'name', 'in_delay'), ...
                        struct('latency', mux_latency), ...
                        {in}, ...
                        {in_delayed});


% add enable signal generator
en = xSignal('en');
en_nondelayed = xSignal('en_nondelayed');
counter = xSignal('counter');
hold_en = xBlock(struct('source', str2func('hold_en_init_xblock'), 'name', 'en_gen'), ...
    {[blk, '/en_gen'], ...
    'hold_period', hold_period, ...
    'counter_out', 'on'}, ...
    {sync}, ...
    {en_nondelayed, counter});
en_delay = xBlock(struct('source', 'Delay', 'name', 'en_delay'), ...
                        struct('latency', mux_latency), ...
                        {en_nondelayed}, ...
                        {en});



num_mux = idivide(int32(len), hold_period);
%  add coefficient constants
coeff_consts = cell(1, len);
coeff_sigs = cell(1,len);
for i=1:num_mux*hold_period
    coeff_sigs{i} = xSignal(['coeff', num2str(len_total-nonzero_ind(i))]);
    coeff_consts{i} = xBlock(struct('source', 'Constant', 'name', ['coefficient',num2str(len_total-nonzero_ind(i) + 1)]), ...
                                struct('const', coeffs(nonzero_ind(i)), ...
                                        'n_bits', n_bits, ...
                                        'bin_pt', bin_pt, ...
                                       'explicit_period', 'on'), ...
                                {}, ...
                                coeff_sigs(i));
end


mux_blks = cell(1, num_mux);
mux_delay_blks = cell(num_mux, hold_period-1);
mux_out = cell(1, num_mux);
tap_out = cell(num_mux, hold_period);
mult_blk = cell(1,num_mux);
mult_out = cell(1, num_mux);
for i = 1:num_mux
   mux_out{i} = xSignal(['mux_out', num2str(i)]);
   mult_out{i} = xSignal(['mult_out', num2str(i)]);
   mux_blks{i} = xBlock(struct('source', 'Mux', 'name', ['Mux', num2str(i)]), ...
              struct('inputs', hold_period, ...
                    'latency', mux_latency), ...
                    {counter, coeff_sigs{hold_period*(i-1)+1:hold_period*i}}, ...
                    mux_out(i));
                
   mult_blk{i} = xBlock(struct('source', 'Mult', 'name', ['Mult',num2str(i)]), ...
                        {}, ...
                        [{in_delayed}, mux_out(i)], ...
                        mult_out(i));
   for j = 1:hold_period
      % coeff_num = len_total - nonzero_ind(hold_period*(i-1)+j) + 1;
       coeff_num = nonzero_ind(hold_period*(num_mux-i)+j + 1);
       tap_out{i,j} = xSignal(['tap_out', num2str(coeff_num)]);
       if j == hold_period
           mux_delay_tmp = hold_period*(coeff_num - 1) + hold_period;
       else
           mux_delay_tmp = hold_period*(coeff_num - 1) + (hold_period-j);
       end
       extra_delay = hold_period*( length(nonzero_ind) - hold_period*num_mux - 1);
       mux_delay_blks{i, j} = xBlock(struct('source', 'Delay', 'name', ['mux_delay',num2str(i), '_', num2str(j)]), ...
                          struct('latency', extra_delay + mux_delay_tmp + hold_period-2), ...  % I don't know why this is so... but multiply with enable seem to have delay like this (hold_fir_tap)
                          mult_out(i), ...
                          tap_out(i, j));
   end
end


hold_fir_tap = cell(1, len-num_mux*hold_period);
hold_fir_tap_out = cell(1, len-num_mux*hold_period);
delay_out = cell(1, len-num_mux*hold_period);
delay_blks = cell(1, len-num_mux*hold_period-1);
for i = 1: (len-num_mux*hold_period)
    coeff_num = nonzero_ind(i+num_mux*hold_period);
    hold_fir_tap_out{i} = xSignal(['hold_fir_tap_out',num2str(i+num_mux*hold_period)]);
    hold_fir_tap{i} = xBlock(struct('source',  str2func('hold_fir_tap_init_xblock'), ...
                                                'name', ['hold_fir_tap',num2str(len_total - coeff_num + 1)]), ...
                              {[blk, '/hold_fir_tap',num2str(len_total - coeff_num + 1)], ...
                                'mult_latency', mult_latency, ...
                                'hold_period', hold_period, ...
                                'coefficient', coeffs(coeff_num), ...
                                'ext_en', 'on'}, ...
                              {in_delayed, en}, ...
                              {hold_fir_tap_out{i}, []});
                          
    if i <len-num_mux*hold_period
        delay_out{i} = xSignal(['delay_out', num2str(i)]);
        delay_blks{i} = xBlock(struct('source', 'Delay', 'name', ['delay',num2str(i)]), ...
                          struct('latency', hold_period*(len_total-coeff_num)), ...
                          hold_fir_tap_out(i), ...
                          delay_out(i));
    else
        delay_out{i} = hold_fir_tap_out{i};
    end
    
end

all_tap_out = cell(1,len);
for i = 1:num_mux
    for j = 1:hold_period
        all_tap_out{hold_period*(i-1)+j} = tap_out{i,j};
    end
end
for i = num_mux*hold_period+1:len
    all_tap_out{i} = delay_out{i-num_mux*hold_period};
end



% add adder tree
adder_tree_delay_en = xSignal('adder_tree_en');
adder_tree_delay = xBlock(struct('source', 'Delay', 'name', 'adder_tree_delay'), ...
                        struct('latency', hold_period+1), ... % I don't know why but...
                        {en}, ...
                        {adder_tree_delay_en});
adder_tree = xBlock(struct('source', str2func('hold_adder_tree_init_xblock'), 'name', 'adder_tree'), ...
    {[blk, '/adder_tree'], ...
    'add_latency', add_latency, ...
    'hold_period', hold_period, ...
    'n_inputs', len, ...
    'ext_en', 'on'}, ...
    {adder_tree_delay_en, all_tap_out{:}}, ...
    {en_out, out});



if ~isempty(blk) && ~strcmp(blk(1),'/')
    clean_blocks(blk);
    fmtstr=sprintf('%d-tap mux-FIR filter (%d nonzero tap)%s\n hold period: %d\n mult latency: %d\n add latency: %d',length(coeffs),len, coeff_msg, hold_period, mult_latency,add_latency);
    set_param(blk,'AttributesFormatString',fmtstr);
end


end

