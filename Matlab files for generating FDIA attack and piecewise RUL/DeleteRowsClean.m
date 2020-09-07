%clean test data so that each engine has atleast 130 time cycles
clc,
clear all;
close all;

%read the test data from the path
N = readtable('PM_test.txt','ReadVariableNames',false);
T= removevars(N,{'Var27'});

%read the ground truth from the path
Q = readtable('PM_truth.txt','ReadVariableNames',false);
R= removevars(Q,{'Var2'});

% define initial engine ID
engineId=1;

% define the engine ID untill which point the data should be cleaned
engineIdEnd=37;

k=1;
g=1;

for i=engineId:engineIdEnd
    Table = T(T.Var1==i,:);
    s=size(Table, 1);
    
    %check if the engine has atleast 150 time cycles
    if s>=150
        coll=k*ones(s,1);
        Table.Var1=coll;
        if(k==1)
           q=Table; 
        end
        if(k>1)
            q=[q;Table];
        end
       k=k+1;
    end
    if s<150
        R.Var1(i)=988;
        g=g+1;
    end
end

R(R.Var1==988,:)=[];

emptyCol = cell(size(q, 1),1);
q=[q emptyCol emptyCol];

emptyCol1 = cell(size(R, 1),1);
R=[R emptyCol1];

k=k-1;
g=g-1;


%write to a text file
writetable(q,'Cleaned_data.txt','WriteVariableNames', false,'Delimiter',' ')
writetable(R,'Cleaned_data_truth.txt','WriteVariableNames', false,'Delimiter',' ') 