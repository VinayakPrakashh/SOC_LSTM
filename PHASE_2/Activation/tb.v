module tb_activation_top;

// Parameters
parameter DATA_WIDTH = 16;
parameter ADDRESS_BITS = 2;

// Testbench signals
reg clk;
reg rst;
reg start;
wire done;

// Instantiate DUT (Device Under Test)
lstm_activation_top #(
    .DATA_WIDTH(DATA_WIDTH),
    .ADDRESS_BITS(ADDRESS_BITS)
) dut (
    .clk(clk),
    .rst(rst),
    .start(start),
    .done(done)
);

// Clock generation (50MHz = 20ns period)
always #10 clk = ~clk;

// Function to convert S7.8 to real for display
function real s7p8_to_real;
    input [15:0] s7p8_val;
    begin
        if (s7p8_val[15]) 
            s7p8_to_real = -((~s7p8_val + 1) / 256.0);
        else 
            s7p8_to_real = s7p8_val / 256.0;
    end
endfunction

// Test stimulus
initial begin
    // Initialize signals
    clk = 0;
    rst = 1;
    start = 0;
    
    $display("=== LSTM Activation Top Testbench ===");
    $display("Testing S7.8 format activation processing");
    $display("");
    
    // Reset sequence
    #20 rst = 0;
    #20;
    
    $display("Time=%0t: Starting activation process...", $time);
    
    // Start activation process
    start = 1;
    #20 start = 0;
    
    // Wait for completion
    wait(done);
    
    $display("Time=%0t: Activation process completed!", $time);
    $display("");
    
    // Display results from all buffers
    $display("=== Final Results ===");
    display_buffer_contents();
    
    #100;
    $display("=== Test Complete ===");
    $finish;
end

// Task to display buffer contents
task display_buffer_contents;
    integer i;
    real input_val, forget_val, cell_val, output_val;
    begin
        $display("Address | Input Gate | Forget Gate | Cell Gate  | Output Gate");
        $display("--------|------------|-------------|------------|------------");
        
        for (i = 0; i < 4; i = i + 1) begin
            // Get values from buffer memories
            input_val = s7p8_to_real(dut.input_buffer.mem_array[i]);
            forget_val = s7p8_to_real(dut.forget_buffer.mem_array[i]);
            cell_val = s7p8_to_real(dut.cell_buffer.mem_array[i]);
            output_val = s7p8_to_real(dut.output_buffer.mem_array[i]);
            
            $display("   %0d    |   %7.4f  |   %7.4f   |  %7.4f   |   %7.4f", 
                     i, input_val, forget_val, cell_val, output_val);
        end
    end
endtask

// Monitor key signals during simulation
initial begin
    $monitor("Time=%0t | Start=%b | Done=%b | State=%b | Counter=%d", 
             $time, start, done, dut.activation_unit.state, dut.activation_unit.counter);
end

// Optional: Display initial values
initial begin
    #50; // Wait for reset to complete
    $display("=== Initial Buffer Values (Before Activation) ===");
    display_buffer_contents();
    $display("");
end

endmodule