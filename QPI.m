function [w, info] = QPI(problem, w, options)

% input:

 % get data sequence
    x = problem.data.x;
    [n,d] = size(x);
    
% get init
    if ~isfield(options, 'maxepoch'); options.maxepoch = 100; end % Maximum number of epochs.
    if ~isfield(options, 'quant'); options.quant = 'none'; end % Quantization type
    if ~isfield(options, 'bits'); options.bits = 4; end % Quantization bits per entry.
    if ~isfield(options, 'num_worker'); options.num_worker = 100; end % number of workers
    if ~isfield(options, 'average_strategy'); options.average_strategy = 'average'; end

    problem_local = rmfield(problem,{'cost','efullgrad'});
    
    opt_local.batchsize = options.batchsize;
    
    num_worker = options.num_worker;
    idx_data_shuffled = randperm(n); % randomly permute the data for assignment to workers
    
    % assign data
    size_data_block = floor(n/num_worker);
    for i = 1:num_worker
        idx_block{i} = idx_data_shuffled(((i-1)*size_data_block+1):(i*size_data_block));
    end
    
    if isempty(w) % if init w is not given, generate it randomly.
        w_init = normrnd(0,1,[d,1]);
        w_current = w_init/norm(w_init);
    else
        w_current = w;
    end
    

    fprintf('-------------------------------------------------------%5d\n',num_worker);
    fprintf('QPI:  epoch\t               cost val\t    upd. norm\n');

        
    upd_worker = zeros(d,num_worker);     
    upd_avg_old = zeros(d,1);
    old = zeros(d,num_worker);

    
    for epoch = 1:options.maxepoch

        cost = problem.cost(w_current);
        info(epoch).loss = cost; 
        fprintf('QPI:  %5d\t%+.16e\n',epoch-1, cost);
        
        for i = 1:num_worker
            upd_worker(:,i) = worker_upd(problem_local,w_current,opt_local,idx_block{i});
        end
        upd_worker_diff = upd_worker-old;

        % compute the quantized average Euclidean updient
        [upd_avg_diff, olddiff] = quantavgdiff(upd_worker_diff,options);    
        old = old + olddiff;

        upd_avg = upd_avg_diff + upd_avg_old;  

        %Do power iteration step
        new_w_upd = upd_avg/norm(upd_avg);
           
        w_current =  new_w_upd;
        
        %update old values
        upd_avg_old = upd_avg;
        
        info(epoch).iterate = w_current;

    end
    
    w = w_current;
    
    
    function [w_upd] = worker_upd(problem,w,options,idx_block)  %Computes the update for a worker
        x_local = x(idx_block,:);   
        perm_idx = randperm(length(idx_block));
        x_perm = x_local(perm_idx,:);
        x_batch = x_perm(1 : options.batchsize,:);   
        w_upd = x_batch'*x_batch*w;
    end

    function [qad,old] = quantavgdiff(upd_worker_diff,options) %Simulates the quantizing, collection, averaging, requantizing, and broadcasting of vectors
        [dim,numworkers] = size(upd_worker_diff);
        qad = zeros(dim,1);
        old = zeros(dim,options.num_worker);
        for i = 1:numworkers
            old(:,i)=quantvector(upd_worker_diff(:,i),options);
            qad = qad + old(:,i)/numworkers;
        end     
        qad = quantvector(qad,options);     %Also returns the vectors' quantized values, since these are needed next round for difference quantization
    end

    function [quant_vec] = quantvector(v,options) %Quantize a given vector
        if strcmp(options.quant,'none') %(If full-precision, just return the vector)
           quant_vec = v;
        else
            amax = max(abs(v)); %Else, use unbiased coordinate-wise rounding procedure
            side = 2*amax / (2^(options.bits-1));
            offset = rand;
            roundv = round(v/side+offset);
            quant_vec = (roundv-offset)*side;
        end
    end


end  
    