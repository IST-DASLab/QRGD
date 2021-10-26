function [w, info] = QPI(problem, w, options)

% input:

 % get data sequence
    x = problem.data.x;
    [n,d] = size(x);
% get init
    if ~isfield(options, 'maxepoch'); options.maxepoch = 100; end % Maximum number of epochs.
    if ~isfield(options, 'stepsize'); options.stepsize = 0.1; end % Initial stepsize (Guess).
    if ~isfield(options, 'quant'); options.quant = 'none'; end % Quantization type
    if ~isfield(options, 'bits'); options.bits = 4; end % Quantization bits per entry.
    if ~isfield(options, 'batchsize'); options.batchsize = n/100; end % Batchsize.
    if ~isfield(options, 'num_worker'); options.num_worker = 100; end % number of workers
    if ~isfield(options, 'maxinneriter'); options.maxinneriter = 0; end % Maximum number of sampling per epoch.
    if ~isfield(options, 'flag_random_shuffle'); options.flag_random_shuffle = 0; end % 
    if ~isfield(options, 'average_strategy'); options.average_strategy = 'average'; end

    
    stepsize = options.stepsize; % for display
    bits= options.bits;

    problem_local = rmfield(problem,{'cost','efullgrad'});
    
    max_inneriter =floor(options.maxinneriter);
    opt_local.maxinneriter = max_inneriter;
    opt_local.batchsize = options.batchsize;
    opt_local.stepsize = stepsize;
    
    num_worker = options.num_worker;
    if options.flag_random_shuffle == 1
        idx_data_shuffled = randperm(n); % random shuffling
    else
        idx_data_shuffled = 1:n; % no shuffling.
    end

    
    % balance data segmentation
    size_data_block = floor(n/num_worker);
    for i = 1:num_worker
        idx_block{i} = idx_data_shuffled(((i-1)*size_data_block+1):(i*size_data_block));
    end
    
    if isempty(w) % if init w is not given, generate it randomly.
        x1_sample = cell2mat(x(idx_block{1}));
        X1 = x1_sample * x1_sample' / size_data_block;
        [V,D] = eig(X1);
        w_outer =  V(:,size(X1,2));
        clear x1_sample X1 V D
    else
        w_outer = w;
    end
    

    fprintf('-------------------------------------------------------%5d\n',num_worker);
    fprintf('CEDRE:  epoch\t               cost val\t    grad. norm\t stepsize\n');

    
% master

    %epstepsize = 1:options.maxepoch;
    %epstepsize = 1.3.^epstepsize / sqrt(d);
    %epstepsize = min(epstepsize,ones(1,options.maxepoch)*opt_local.stepsize);
        
    grad_worker_old = zeros(d,num_worker);
    grad_worker = zeros(d,num_worker);     
    grad_worker_diff = zeros(d,num_worker); 
    grad_avg_old = zeros(d,1);
    grad_avg = zeros(d,1);
    grad_avg_diff = zeros(d,1); 
    old = zeros(d,num_worker);
    rgrad_worker_old = zeros(d-1,num_worker);
    rgrad_worker = zeros(d-1,num_worker);     
    rgrad_worker_diff = zeros(d-1,num_worker);    
    
    
 
    rgrad_avg = zeros(d-1,1);

    
    for epoch = 1:options.maxepoch
        
        egrad = problem.efullgrad(w_outer);
        rgrad_full = problem.M.egrad2rgrad(w_outer,egrad);
        
        gradnorm = problem.M.norm(w_outer, rgrad_full);
        info(epoch).gradnorm = gradnorm;
        cost = problem.cost(w_outer);
        info(epoch).loss = cost; 
        fprintf('CEDRE:  %5d\t%+.16e\t%.8e\n',epoch-1, cost, gradnorm);
        
        for i = 1:num_worker
            grad_worker(:,i) = worker_upd(problem_local,w_outer,opt_local,idx_block{i});
        end
        grad_worker_diff = grad_worker-old;

        % compute the quantized average Euclidean gradient
        [grad_avg_diff, olddiff] = quantavgdiff(grad_worker_diff,options);    
        old = old + olddiff;

        grad_avg = grad_avg_diff + grad_avg_old;  

        %Do power iteration step
        new_w_grad = grad_avg/norm(grad_avg);
           
        w_outer =  new_w_grad;
        
        %update old values
        grad_avg_old = grad_avg;
        grad_worker_old = grad_worker;
        
        info(epoch).iterate = w_outer;

    end
    
    w = w_outer;
    
    
    
    function [w_upd] = worker_upd(problem,w,options,idx_block)
        % load local data
        x_local = x(idx_block,:);   
        perm_idx = randperm(length(idx_block));
        x_perm = x_local(perm_idx,:);
        x_batch = x_perm(1 : options.batchsize,:);   
        w_upd = x_batch'*x_batch*w;
    end


    function [qad,old] = quantavgdiff(grad_worker_diff,options)
        [dim,numworkers] = size(grad_worker_diff);
        qad = zeros(dim,1);
        old = zeros(dim,numworkers);
        for i = 1:numworkers
            old(:,i)=quantvector(grad_worker_diff(:,i),options);
            qad = qad + old(:,i)/numworkers;
        end     
        qad = quantvector(qad,options);
    end

    function [quant_vec] = quantvector(v,options)
        if strcmp(options.quant,'none')
           quant_vec = v;
        else
            amax = max(abs(v));
            side = 2*amax / (2^(options.bits-1));
            offset = rand;
            roundv = round(v/side+offset);
            quant_vec = (roundv-offset)*side;
        end
    end


end  
    