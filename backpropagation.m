function [dW,dB] = backpropagation(input,W,B,likelihood,batch,category,nsize,dropout)

train = input(batch,:);
label = category(batch,:);
prob = likelihood(batch,:);
l = length(nsize);
dW = cell(1,l-1);
dB = cell(1,l-1);

for i = 1:1:l-1
    dW{i} = 0*W{i};
    dB{i} = 0*B{i};
end

Backward = cell(1,l-1);

for i = 1:1:length(batch)    
    Forward{1} = train(i,:)';   
    for k = 1:1:l-1                                     
        if (k == l-1)
            Forward{k+1} = softmax(Forward{k},W{k},B{k});
        else
            [Forward{k+1},mask{k}] = feedforward(Forward{k},W{k},B{k},dropout);
        end        
    end
        
    f = Forward{k+1}';
    class = zeros(size(prob,2),1)';
    class(label(i)+1) = 1;
    Backward{end} = -(class -f)';   
    
    for j=1:1:l-2
        Backward{end-j} = mask{end-j+1}.*((W{end-j+1}'*Backward{end-j+1}).*(Forward{end-j}.*(1-Forward{end-j})));
    end
    
    for j = 1:1:l-1        
        dB{j} = dB{j} + Backward{j};        
        dW{j} = dW{j} + Backward{j}*Forward{j}';
    end
    
end


end