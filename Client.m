classdef Client
    properties
        tcpipClient
    end
    
    methods
        function obj = Client(port)
%           obj.tcpipClient = tcpip('172.20.28.50', port, 'NetworkRole', 'Client');
        %  obj.tcpipClient = tcpip('192.168.0.109', port, 'NetworkRole', 'Client');
        obj.tcpipClient = tcpip('192.168.1.11', port, 'NetworkRole', 'Client');
            %obj.tcpipClient = tcpip('127.0.0.1', port, 'NetworkRole', 'Client');
            set(obj.tcpipClient,'Timeout', 10);
        end

        function send(obj, value)
            fopen(obj.tcpipClient);
            
            fprintf(obj.tcpipClient, '%e ', value);
            
            fclose(obj.tcpipClient);
        end 
    end
end