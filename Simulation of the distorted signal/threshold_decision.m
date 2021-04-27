
T = readtable('predicted 30.csv');
X=table2array(T);
xx=2e6;

model_output(1)=4;
for i=2:xx+1
    
        if X(i-1)<=0.5
        model_output(i,:)=0;
        else 
        model_output(i,:)=1;
        end
    
    
end

MMMM=[model_output];
dlmwrite('final output after the signal recovery 30.csv',MMMM,'-append');