
load humanactivity.mat;
A = importdata('spambase.data');
B=importdata('Data_Cortex_Nuclear.xls').data;
B(isnan(B))=0;
C=feat;
D=importdata('movement_libras.data');
D(isnan(B))=0;
[w,info] = runQPCA1(C);

function [w,info] = runQPCA1(dataset)
% function [w,info] =runCEDRE(input_filename,opt)

% setting of opt:
% maxepoch: max global iterations
% tolgradnorm: tolerance of full gradient norm
% stepsize: step size
% batchsize: batch size
% num_worker: number of workers
% maxinneriter: max local iterations
% flag_random_shuffle: Set true to randomly shuffle the data
% average_strategy: 'random'-random selection; 'average'-sign-fixed average
%   'geo-avg'-geodesically averaging

    input_filename = 'examgrades.mat';
    %input_data = examgrades;
    %input_data.x_sample = normrnd(0, 1, 1000,200);
    
    x_sample = dataset;

    %x_sample = eye(1000);
    %x_sample(1,1) = 2;
    [n,d] = size(x_sample);
    disp(size(x_sample));
    % data
    data.x = x_sample;
    problem.data = data;

    % Create the problem structure.
    manifold = spherefactory(d);
    problem.M = manifold;

    % Define the problem cost function and its Euclidean gradient.
    problem.cost = @cost;
    function f = cost(w)
        f = - w' * x_sample' * x_sample * w;
        f = f/n;
    end

    problem.egrad = @egrad;
    function g = egrad(w,x)
        g =  -2*x'*(x*w);
    end

    problem.efullgrad = @efullgrad;
    function g = efullgrad(w)
        g = - 2* x_sample'*(x_sample*w);
    end

    problem.egrad_batchsize = @egrad_batchsize;
    function g = egrad_batchsize(U, data_batchsize)
        x_sample_batchsize = cell2mat(data_batchsize);        
        N_batchsize = size(x_sample_batchsize, 2);
        g =  -x_sample_batchsize*(x_sample_batchsize'*U);
        g = g/N_batchsize;
    end
                               %Now just need to average over runs
    options=[];
    options.maxepoch=31;
    options.stepsize =1e-9;   %1e-9 for A 1 to 5 e-5 for B, 1e-9 for C, 1e-4 for D 
    options.num_worker = 5;
    options.batchsize = int32(n/options.num_worker);
    options.maxinneriter = 1;
    options.flag_random_shuffle = 0;
    options.bits=10;
    options.lrtype = 'const';
    
    w_init = normrnd(0,1,[d,1]);
    w_init = w_init/norm(w_init);
    
    runs = 10;
    
    sum0=zeros(1,options.maxepoch);
    sum1=zeros(1,options.maxepoch);    
    sum2=zeros(1,options.maxepoch);
    sum3=zeros(1,options.maxepoch);    
    
    for i=1:runs
        options.quant = 'none';    

        [w, info] = QPCA(problem, w_init, options);
        sum0=sum0+[info.loss];
        
        options.quant = 'quant';  
        [w1, info1] = QPCA(problem, w_init, options);
        sum1=sum1+[info1.loss];
        
        options.quant = 'rquant';  
        [w2, info2] = QPCA(problem, w_init, options);
        sum2=sum2+[info2.loss];       
        
        options.quant = 'quant';  
        [w3, info3] = QPI(problem, w_init, options);
        
        %options.bits=8;
        %start = info3(1).iterate;
        %[w3, info3] = QPCA(problem, start, options);
        sum3=sum3+[info3.loss];
    end
    
    avg0 = sum0 / runs;
    avg1 = sum1 / runs;
    avg2 = sum2 / runs;
	avg3 = sum3 / runs;   
    
    plot([0:options.maxepoch-1],[avg0],[0:options.maxepoch-1],[avg1],[0:options.maxepoch-1],[avg2],[0:options.maxepoch-1],[avg3]);
    legend('Full precision Riemannian GD','Euclidean gradient difference quantization','Riemannian gradient quantization','Quantized power iteration');
    title({'Comparison of quantization methods for Riemannian gradient descent' ,strcat('Human Activity,',' m = ',num2str(n),' d = ',num2str(d),' eta = ',num2str(options.stepsize), ' n = ',num2str(options.num_worker),' bits = ',num2str(options.bits))});
    xlabel('Epoch');
    ylabel('Cost -x^{T}Ax');
    
end