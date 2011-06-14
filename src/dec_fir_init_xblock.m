function dec_fir_init_xblock(n_inputs, coeff, n_bits, quantization, add_latency, mult_latency, coeff_bit_width, coeff_bin_pt, adder_tree_hdl)

%% initialization scripts
% round coefficients to make sure rounding error doesn't prevent us from
% detecting symmetric coefficients
coeff_round = round(coeff * 1e16) * 1e-16;

if mod(length(coeff)/n_inputs,1) ~= 0,
    error('The number of coefficients must be integer multiples of the number of inputs');
end

num_fir_col = length(coeff)/n_inputs;
disp(num_fir_col);
if coeff_round(1:length(coeff)/2) == coeff_round(length(coeff):-1:length(coeff)/2+1),
    num_fir_col = num_fir_col / 2;
    fir_col_type = 'fir_dbl_col'
    coeff_sym = 1;
else
    fir_col_type = 'fir_col'
    coeff_sym = 0;
end



%% inports
xlsub2_sync_in = xInport('sync_in');
xlsub2_real=cell(n_inputs,1);
xlsub2_imag=cell(n_inputs,1);
for i = 1:n_inputs,
    xlsub2_real{i} = xInport(['real',num2str(i)]);
    xlsub2_imag{i} = xInport(['imag',num2str(i)]);
end

%% outports
xlsub2_sync_out = xOutport('sync_out');
xlsub2_dout = xOutport('dout');

%% diagram


%delay of sync
if coeff_sym,
    % y(n) = sum(aix(n-i)) for i=0:N. sync is thus related to x(0)
    sync_latency = add_latency + mult_latency + ceil(log2(n_inputs))*add_latency + add_latency;
%    sync_latency = 2*num_fir_col + mult_latency + ceil(log2(num_fir_col))* add_latency + 2*add_latency;
else
    sync_latency = mult_latency + ceil(log2(n_inputs))*add_latency + add_latency;
%    sync_latency = (num_fir_col-1) + mult_latency + ceil(log2(n_inputs))*add_latency + ceil(log2(num_fir_col) + add_latency;
end
% block: untitled/dec_fir/delay
xlsub2_delay_out1 = xSignal;
xlsub2_delay = xBlock(struct('source', 'Delay', 'name', 'delay'), ...
                      struct('latency', sync_latency), ...
                      {xlsub2_sync_in}, ...
                      {xlsub2_delay_out1});
                  
xlsub2_real_sum_inports=cell(1,num_fir_col+2);
xlsub2_real_sum_out2 = xSignal;
% for i=2:num_fir_col+2,
%     xlsub2_real_sum_inports{i}=xSignal(['xlsub2_real_sum_inports',num2str(i)]);
% end
xlsub2_real_sum_inports{1}=xlsub2_delay_out1;


xlsub2_imag_sum_inports=cell(1,num_fir_col+2);
xlsub2_imag_sum_out2 = xSignal;
% for i=2:num_fir_col+2,
%     xlsub2_imag_sum_inports{i}=xSignal(['xlsub2_imag_sum_inports',num2str(i)]);
% end
xlsub2_imag_sum_inports{1}=xlsub2_delay_out1;




% block: fir_col or fir_dbl_col       
if coeff_sym
    sym = 4;
    num_fir_col_sym = num_fir_col*2;
else
    sym = 2;
    num_fir_col_sym = num_fir_col +1;
end

xlsub2_fir_col=cell(num_fir_col_sym,n_inputs*sym+2);

xlsub2_fir_col_sub=cell(1,num_fir_col);

% initiliaze signal array
for j=1:n_inputs,
    xlsub2_fir_col{1,2*j-1} = xlsub2_real{j};
    xlsub2_fir_col{1,2*j} = xlsub2_imag{j};
end
for j=(2*n_inputs+1): n_inputs*sym+2,
    xlsub2_fir_col{1,j} = xSignal(['xlsub2_fir_col',num2str(1),'_',num2str(j)]);
end
for i = 1:num_fir_col_sym,
    for j=1:n_inputs*sym+2,
        xlsub2_fir_col{i+1,j}=xSignal(['xlsub2_fir_col',num2str(i),'_',num2str(j)]);
    end
end



if coeff_sym,
    for k=1:n_inputs,
        xlsub2_fir_col{num_fir_col*2-1,2*n_inputs+k*2-1}=xlsub2_fir_col{num_fir_col*2,k*2-1};
        xlsub2_fir_col{num_fir_col*2-1,2*n_inputs+k*2}=xlsub2_fir_col{num_fir_col*2,k*2};
    end
end


for i=1:num_fir_col,
    % block: untitled/dec_fir/fir_col1
    

    if coeff_sym,
        if i< num_fir_col,
            for k=1:n_inputs,
                xlsub2_fir_col{i*2-1,2*n_inputs+k*2-1}=xlsub2_fir_col{i*2+2,2*n_inputs+k*2-1};
                xlsub2_fir_col{i*2-1,2*n_inputs+k*2}=xlsub2_fir_col{i*2+2,2*n_inputs+k*2};
            end
            
            for j=1:n_inputs,
                xlsub2_fir_col{i*2+1,j*2-1}=xlsub2_fir_col{i*2,j*2-1};
                xlsub2_fir_col{i*2+1,j*2}=xlsub2_fir_col{i*2,j*2};
            end
        end
       


        
        xlsub2_real_sum_inports{i+1}=xlsub2_fir_col{i*2,n_inputs*sym+1};
        xlsub2_imag_sum_inports{i+1}=xlsub2_fir_col{i*2,n_inputs*sym+2};
    else
        xlsub2_real_sum_inports{i+1}=xlsub2_fir_col{i+1,n_inputs*sym+1};
        xlsub2_imag_sum_inports{i+1}=xlsub2_fir_col{i+1,n_inputs*sym+2};
    end
    
    
    if coeff_sym,
        xlsub2_fir_col_sub{i} = xBlock(struct('source',str2func([fir_col_type,'_init_xblock']), 'name', [fir_col_type,num2str(i)]), ...
                                 {n_inputs, coeff(i*n_inputs:-1:(i-1)*n_inputs+1), ...
                                   mult_latency,add_latency, ...
                                   coeff_bit_width,coeff_bin_pt, adder_tree_hdl},...
                                 {xlsub2_fir_col{i*2-1,1:sym*n_inputs}}, ...
                                 {xlsub2_fir_col{i*2,1:sym*n_inputs+2}});
    else
        xlsub2_fir_col_sub{i} = xBlock(struct('source',str2func([fir_col_type,'_init_xblock']), 'name', [fir_col_type,num2str(i)]), ...
                                 {n_inputs, coeff(i*n_inputs:-1:(i-1)*n_inputs+1), ...
                                   mult_latency,add_latency, ...
                                   coeff_bit_width,coeff_bin_pt, adder_tree_hdl},...
                                 {xlsub2_fir_col{i,1:sym*n_inputs}}, ...
                                 {xlsub2_fir_col{i+1,1:sym*n_inputs+2}});        
    end
end

% block: untitled/dec_fir/imag_sum
xlsub2_imag_sum_sub = xBlock(struct('source',str2func('adder_tree_init_xblock'), 'name', 'imag_sum'), ...
                         {num_fir_col+1, add_latency, 'off','off'}, ...   %% for now just set them to off, ignore 'adder_tree_hdl'
                         xlsub2_imag_sum_inports, ...
                         {[], xlsub2_imag_sum_out2});

% block: untitled/dec_fir/real_sum
xlsub2_real_sum_sub = xBlock(struct('source', str2func('adder_tree_init_xblock'), 'name', 'real_sum'), ...
                         {num_fir_col+1, add_latency,'off','off'}, ... %% for now just set them to off, ignore 'adder_tree_hdl'
                         xlsub2_real_sum_inports, ...
                         {xlsub2_sync_out, xlsub2_real_sum_out2});
                     
                     
% block: untitled/dec_fir/convert1
xlsub2_shift1_out1 = xSignal;
xlsub2_convert1_out1 = xSignal;
xlsub2_convert1 = xBlock(struct('source', 'Convert', 'name', 'convert1'), ...
                         struct('n_bits', n_bits, ...
                                'bin_pt', n_bits-1, ...
                                'arith_type', 'Signed  (2''s comp)', ...
                                'quantization', quantization,...
                                'latency', add_latency), ...
                         {xlsub2_shift1_out1}, ...
                         {xlsub2_convert1_out1});

% block: untitled/dec_fir/convert2
xlsub2_shift2_out1 = xSignal;
xlsub2_convert2_out1 = xSignal;
xlsub2_convert2 = xBlock(struct('source', 'Convert', 'name', 'convert2'), ...
                         struct('n_bits', n_bits, ...
                                'bin_pt', n_bits-1, ...
                                'arith_type', 'Signed  (2''s comp)', ...
                                'quantization', quantization,...
                                'latency', add_latency), ...
                         {xlsub2_shift2_out1}, ...
                         {xlsub2_convert2_out1});



% block: untitled/dec_fir/ri_to_c
xlsub2_ri_to_c = xBlock(struct('source', 'casper_library_misc/ri_to_c', 'name', 'ri_to_c'), ...
                        [], ...
                        {xlsub2_convert1_out1, xlsub2_convert2_out1}, ...
                        {xlsub2_dout});

% block: untitled/dec_fir/shift1
xlsub2_shift1 = xBlock(struct('source', 'Shift', 'name', 'shift1'), ...
                       struct('shift_bits', 1,'shift_dir', 'Left'), ...
                       {xlsub2_real_sum_out2}, ...
                       {xlsub2_shift1_out1});

% block: untitled/dec_fir/shift2
xlsub2_shift2 = xBlock(struct('source', 'Shift', 'name', 'shift2'), ...
                       struct('shift_bits', 1,'shift_dir', 'Left'), ...
                       {xlsub2_imag_sum_out2}, ...
                       {xlsub2_shift2_out1});

% 
% if coeff_sym,
%     for h=1:num_fir_col,
%         if h ~= 1
%             for k=1:n_inputs,
%                 xlsub2_fir_col{h,2*n_inputs+k*2-1}=xlsub2_fir_col{h+1,2*n_inputs+k*2-1};
%                 xlsub2_fir_col{h,2*n_inputs+k*2}=xlsub2_fir_col{h+1,2*n_inputs+k*2};
%             end
%         end
%     end
% 
%     for k=1:n_inputs,
%         xlsub2_fir_col{num_fir_col+1,2*n_inputs+k*2-1}=xlsub2_fir_col{num_fir_col+1,k*2-1};
%         xlsub2_fir_col{num_fir_col+1,2*n_inputs+k*2}=xlsub2_fir_col{num_fir_col+1,k*2};
%     end
% end



end



