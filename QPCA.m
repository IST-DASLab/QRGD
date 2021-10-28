function [w, info] = QPCA(problem, w, options)

% input:

 % get data sequence
    x = problem.data.x;
    [n,d] = size(x);
% get init
    if ~isfield(options, 'maxepoch'); options.maxepoch = 100; end % Maximum number of epochs.
    if ~isfield(options, 'lrtype'); options.lrtype = 'accel'; end % Maximum number of epochs.
    if ~isfield(options, 'stepsize'); options.stepsize = 0.1; end % Initial stepsize (Guess).
    if ~isfield(options, 'quant'); options.quant = 'none'; end % Quantization type
    if ~isfield(options, 'bits'); options.bits = 4; end % Quantization bits per entry.
    if ~isfield(options, 'batchsize'); options.batchsize = n/100; end % Batchsize.
    if ~isfield(options, 'num_worker'); options.num_worker = 100; end % number of workers
    if ~isfield(options, 'average_strategy'); options.average_strategy = 'average'; end

    
    stepsize = options.stepsize; 
    problem_local = rmfield(problem,{'cost','efullgrad'});
    
    opt_local.batchsize = options.batchsize;
    opt_local.stepsize = stepsize;
    
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
    fprintf('QPCA:  epoch\t               cost val\t    grad. norm\t stepsize\n');
    
    if options.lrtype == 'const'
        epstepsize = ones(options.maxepoch)*options.stepsize;
    elseif options.lrtype == 'accel'    %Accelerated learning-rate schedule, as suggested by theory - recommend using const in practice
        B = options.maxepoch/sqrt(d);
        steps = 1:options.maxepoch;
        steps = min(steps, ones(1,options.maxepoch)*(options.maxepoch-B));
        epstepsize = ones(1,options.maxepoch)*(options.maxepoch+B); 
        epstepsize = epstepsize-steps;
        epstepsize = ones(1,options.maxepoch)./epstepsize;
        epstepsize=epstepsize*options.stepsize;
    end    
        
    grad_worker = zeros(d,num_worker);     
    grad_avg_old = zeros(d,1);
    grad_avg = zeros(d,1);
    old = zeros(d,num_worker);
    rgrad_worker = zeros(d-1,num_worker);     

   
    for epoch = 1:options.maxepoch
        
        egrad = problem.efullgrad(w_current);
        rgrad_full = problem.M.egrad2rgrad(w_current,egrad);
        gradnorm = problem.M.norm(w_current, rgrad_full);
        info(epoch).gradnorm = gradnorm;
        cost = problem.cost(w_current);
        info(epoch).loss = cost; 
        fprintf('QPCA:  %5d\t%+.16e\t%.8e\t%.8e\n',epoch-1, cost, gradnorm, epstepsize(epoch));
        
        %if quantization is in Euclidean space:
        if ~strcmp(options.quant,'rquant')
            % call local worker to perform local computation
            for i = 1:num_worker
                grad_worker(:,i) = worker_grad(problem_local,w_current,opt_local,idx_block{i});
            end
            grad_worker_diff = grad_worker-old;
        
            % compute the quantized average Euclidean gradient
            [grad_avg_diff, olddiff] = quantavgdiff(grad_worker_diff,options);    
            old = old + olddiff;
            
            grad_avg = grad_avg_diff + grad_avg_old;  

            %convert to riemannian & take step 
            new_w_grad = problem.M.egrad2rgrad(w_current,grad_avg);      
            
        else %quantization is in tangent space
            obasis = tangentorthobasis(problem.M, w_current);     
            for i = 1:num_worker
                workergrad = worker_grad(problem_local,w_current,opt_local,idx_block{i});
                worker_rgrad = problem.M.egrad2rgrad(w_current,workergrad);
                vec = tangent2vec(problem.M, w_current ,obasis, worker_rgrad);
                rgrad_worker(:,i) = vec;
            end
            
            %quantize & avg in tangent space
            rgrad_avg = quantavgdiff(rgrad_worker,options);  
            
            %convert back to full euclidean
            new_w_grad = lincomb(problem.M, w_current,obasis, rgrad_avg);            

        end
        
        w_current =  problem.M.exp(w_current, new_w_grad, -epstepsize(epoch));   

        %rescale to norm 1 to deal with rounding errors within M.exp()
        w_current = w_current/norm(w_current);
        
        %update old values
        grad_avg_old = grad_avg;
        grad_worker_old = grad_worker;

    end
    w = w_current;
    
    
    
    function [w_grad] = worker_grad(problem,w,options,idx_block)
        x_local = x(idx_block,:);   
        perm_idx = randperm(length(idx_block));
        x_perm = x_local(perm_idx,:);
        x_batch = x_perm(1 : options.batchsize,:);  %Pick a random sample of size batchsize from the worker's local data
        w_grad = problem.egrad(w,x_batch);  %return the Euclidean gradient on that sample
    end


    function [qad,old] = quantavgdiff(grad_worker_diff,options) %Simulates the quantizing, collection, averaging, requantizing, and broadcasting of vectors
        [dim,numworkers] = size(grad_worker_diff);
        qad = zeros(dim,1);
        old = zeros(dim,options.num_worker);
        for i = 1:numworkers
            old(:,i)=quantvector(grad_worker_diff(:,i),options);
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
    