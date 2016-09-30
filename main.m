clc
clear
close all

% Cd to folder of code
if(~isdeployed)
    cd(fileparts(which(mfilename)));
end

% Read file
file1 = dlmread('digitstrain.txt',',');
train = file1(:,1:end-1);
label = file1(:,end);
file2 = dlmread('digitsvalid.txt',',');
valid = file2(:,1:end-1);
labelv = file2(:,end);
file3 = dlmread('digitstest.txt',',');
test = file3(:,1:end-1);
labelt = file3(:,end);

% Tunable parameters
eta = 0.1;                                       % Learning Rate
moment = 0;                                     % Momentum
epochs = 25;                                      % Early Stop
output_size = 10;                                 % Number of output labels
hsize = [100];                                 % Number of weights in each hidden layer
dropout = 0;                                      % Dropout probability
lambda = 0;                                    % Regularization

% Network size
input_size = size(train,2);
sample_size = size(train,1);
valid_size = size(valid,1);
test_size = size(test,1);
nsize = [input_size hsize output_size];
l = length(nsize);

% Arrays for storing accuracy metrics
acc = [];
accuracy = [];
accv = [];
acct = [];
accuracyv = [];
accuracyt = [];
meanerr = [];
meanerrv = [];
meanerrt = [];


% Initialization of weights and biases
for i = 1:1:(l-1)
    b = sqrt(6)/sqrt(nsize(i)+nsize(i+1));
    W{i} = -b + 2*b*rand(nsize(i+1),nsize(i));
    B{i} = zeros(nsize(i+1),1);
end

for j = 1:1:sample_size                                 % Over all input examples
    Forward{1} = train(j,:)';
    for k = 1:1:l-1                                     % Layer by layer
        if (k == l-1)
            Forward{k+1} = softmax(Forward{k},W{k},B{k});
        else
            [Forward{k+1}, mask{k}] = feedforward(Forward{k},W{k},B{k},dropout);
        end
    end
    likelihood(j,:) = Forward{end};
end
err = crossentropy(label,likelihood);

% Running the epochs
for i = 1:1:epochs                                         % Number of epochs
    r = randperm(sample_size);
    batch_size = 1;
    n = ceil(sample_size/batch_size);
    
    for k = 1:1:l-1
        pW{k} = 0*W{k};
        pB{k} = 0*B{k};
    end
    
    for t=1:1:n
        if (t < n)
            batch = r((t-1)*batch_size+1:t*batch_size);
        else
            batch = r((t-1)*batch_size+1:end);
        end
        
        [dW, dB] = backpropagation(train,W,B,likelihood,batch,label,nsize,dropout);
        
        for k = 1:1:l-1
            W{k} = W{k} - eta/batch_size*(dW{k} + moment*pW{k} + lambda*W{k});
            B{k} = B{k} - eta/batch_size*(dB{k} + moment*pB{k} + lambda*B{k});
        end
        pW = dW;
        pB = dB;
    end
    
 %%   
    for j = 1:1:sample_size                                 % Over all input examples
        Forward{1} = train(j,:)';
        for k = 1:1:l-1                                     % Layer by layer
            if (k == l-1)
                Forward{k+1} = softmax(Forward{k},W{k},B{k});
            else
                [Forward{k+1},mask{k}] = feedforward(Forward{k},W{k},B{k},0);
            end
        end
        likelihood(j,:) = Forward{end};
    end
    
    err = crossentropy(label,likelihood);
    acc = [acc err/sample_size];
    
    temp = likelihood';
    [M , index] = max(temp);
    index = index-1;
    index = index';
    corr = (index == label);
    accuracy = [accuracy sum(corr)];
    meanerr = [meanerr 100*(sample_size-sum(corr))/sample_size];
    
 %%   
    for j = 1:1:valid_size                                  % Over all input examples
        Forward{1} = valid(j,:)';
        for k = 1:1:l-1                                     % Layer by layer
            if (k == l-1)
                Forward{k+1} = softmax(Forward{k},W{k},B{k});
            else
                [Forward{k+1}, mask{k}] = feedforward(Forward{k},W{k},B{k},0);
            end
        end
        likelihoodv(j,:) = Forward{end};
    end
    
    errv = crossentropy(labelv,likelihoodv);
    accv = [accv errv/valid_size];
    temp = likelihoodv';
    [M , indexv] = max(temp);
    indexv = indexv-1;
    indexv = indexv';
    corrv = (indexv == labelv);
    accuracyv = [accuracyv sum(corrv)];  
    meanerrv = [meanerrv 100*(valid_size-sum(corrv))/valid_size];
    
%%   
    for j = 1:1:test_size                                  % Over all input examples
        Forward{1} = test(j,:)';
        for k = 1:1:l-1                                     % Layer by layer
            if (k == l-1)
                Forward{k+1} = softmax(Forward{k},W{k},B{k});
            else
                [Forward{k+1}, mask{k}] = feedforward(Forward{k},W{k},B{k},0);
            end
        end
        likelihoodt(j,:) = Forward{end};
    end
    
    errt = crossentropy(labelt,likelihoodt);
    acct = [acct errt/test_size];
    
    temp = likelihoodt';
    [M , indext] = max(temp);
    indext = indext-1;
    indext = indext';
    corrt = (indext == labelt);
    accuracyt = [accuracyt sum(corrt)];      
    meanerrt = [meanerrt 100*(test_size-sum(corrt))/test_size];
end

figure(1)
t = (1:epochs);
plot(t,acc,t,accv);
title('Average Cross-Entropy Training (blue) & Validation (red) Error');
ylabel('Error');
xlabel('Epoch');

figure(2)
t = (1:epochs);
plot(t,meanerr,t,meanerrv);
title('Mean Classification Training (blue) & Validation (red) Error');
ylabel('Error');
xlabel('Epoch');

figure(3)
Wt = W{1};
siz = (size(Wt,1));
k = ceil(sqrt(siz));
for j = 1:1:size(Wt,1)
    img = Wt(j,:);
    pic = reshape(img,28,28);
    subplot(k,k,j);
    imagesc(pic');
end









