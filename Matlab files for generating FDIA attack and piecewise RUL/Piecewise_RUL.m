clc,
clear all;
close all;

%Path for Test data to generate piece-wise RUL
N = readtable('myDataN2.txt','ReadVariableNames',false);
T= removevars(N,{'Var27'});

%Path for True RUL for the test data mentioned above
Q = readtable('myDataTrue.txt','ReadVariableNames',false);
R= removevars(Q,{'Var2'});

%specify the engine ID
engineId=17;

Table=T(T.Var1==engineId,:);
s=size(Table, 1);
RUL= R.Var1(engineId);
k=RUL+s;

for i=1:s
    ta=Table(Table.Var2<=i,:);
    if(i==1)
           coll=i*ones(i,1);
           ta.Var1=coll;
           q=ta; 
           R.Var1(i)=k;
    end
    if(i>1)
          coll=i*ones(i,1);
          ta.Var1=coll;
          q=[q;ta]; 
          R.Var1(i)=k;
    end
    k=k-1;
end

emptyCol = cell(size(q, 1),1);
q=[q emptyCol emptyCol];

emptyCol1 = cell(size(R, 1),1);
R=[R emptyCol1];

%write to a text file
writetable(q,'PiecewiseData.txt','WriteVariableNames', false,'Delimiter',' ')

%write RUL to a text file
writetable(R,'PiecewiseRUL.txt','WriteVariableNames', false,'Delimiter',' ') 