function plot_PCA(training_set,labels_training)
  %Computing the principal components for each class
                    [labels_training,I]=sort(labels_training,'ascend');
                    training_set=training_set(I,:);
                    
                    label=unique(labels_training);
                    [coeff,score]= pca(double(training_set));
                     [n_score,m_score]=size(training_set);
                     [np,mp]=size(label);
                     figure;%(filter_type*kcycle*10+filter_type*kcycle);
                     
                        color_defined=[1 0 0;0 1 0;0 0 1;1 1 0;1 0 1;0 1 1];
                     if np>6
                        color_defined=[color_defined;rand((np-6)^2,3,'single')]; 
                     end
                     I=[];
                     I1=[];
                     for ip=1:np
                       %  I=sort([I;find(targets_out{1}(:,1)==cell2mat(text_label_class(ip,1)))]); %find(targets_out{1}(:,1)==72 | targets_out{1}(:,1)==73 | targets_out{1}(:,1)==74 | targets_out{1}(:,1)==5);
                       %  I1=sort([I1;find(targets_out{2}(:,1)==cell2mat(text_label_class(ip,1)))]);%find(targets_out{2}(:,1)==72 | targets_out{2}(:,1)==73 | targets_out{2}(:,1)==74 | targets_out{2}(:,1)==5);
                         data_class=find(labels_training==label(ip,1));
                     
                          if m_score>=3

                              scatter3(score(data_class,1),score(data_class,2),score(data_class,3),'o','MarkerEdgeColor',color_defined(ip,:),'MarkerFaceColor','none');hold on;
                              xlabel('first component');
                              ylabel('second component');
                              zlabel('third component');
                          else
                              scatter(score(data_class,1),score(data_class,2),'o','MarkerEdgeColor',color_defined(ip,:),'MarkerFaceColor','none');hold on;
                               xlabel('first component');
                               ylabel('second component');
                          end
                     
                     
                     end
                    %  total_patterns=length(I1);
                         
                      legend(num2str(label),'Location','Best');
                      set(gca,'Color','none'); %,'Projection','Perspective');


end

