// Example: How to use the weights.mem file in Verilog

module lstm_weights_memory (
    input wire clk,
    input wire [15:0] addr,        // Address: 0 to 37599
    output reg [15:0] weight_data  // Weight value (S7.8)
);

    // Memory array: 37600 words x 16 bits
    reg [15:0] weight_mem [0:37599];
    
    // Load weights from file
    initial begin
        $readmemh("weights_376x100.mem", weight_mem);
        $display("Loaded %0d weights", 37600);
    end
    
    // Read weight
    always @(posedge clk) begin
        weight_data <= weight_mem[addr];
    end
    
endmodule

// Example: Access specific weight
module lstm_weight_access_example;
    
    reg [15:0] weight_mem [0:37599];
    
    initial begin
        $readmemh("weights_376x100.mem", weight_mem);
        
        // Access weight at row=0, col=0
        $display("Weight[0][0] = %h", weight_mem[0]);
        
        // Access weight at row=5, col=10
        // Address = row * 100 + col = 5 * 100 + 10 = 510
        $display("Weight[5][10] = %h", weight_mem[510]);
        
        // Extract sign and magnitude
        automatic logic sign;
        automatic logic [14:0] magnitude;
        
        sign = weight_mem[0][15];
        magnitude = weight_mem[0][14:0];
        
        $display("Sign bit: %b, Magnitude: %h", sign, magnitude);
    end
    
endmodule
