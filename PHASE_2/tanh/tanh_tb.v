module tb_tanh_simple;
    
    // Testbench signals
    reg [15:0] input_value;
    wire [15:0] tanh_out;
    
    // Instantiate DUT
    tanh dut (
        .input_value(input_value),
        .tanh_out(tanh_out)
    );
    
    // Function to convert S7.8 to real
    function real s7p8_to_real;
        input [15:0] s7p8_val;
        begin
            if (s7p8_val[15]) 
                s7p8_to_real = -((~s7p8_val + 1) / 256.0);
            else 
                s7p8_to_real = s7p8_val / 256.0;
        end
    endfunction
    
    // Simple test task
    task test;
        input [15:0] test_val;
        input [63:0] description;
        real input_real, output_real;
        begin
            input_value = test_val;
            #1;
            input_real = s7p8_to_real(test_val);
            output_real = s7p8_to_real(tanh_out);
            $display("%s: Input=0x%04X (%.3f) ? Output=0x%04X (%.3f)", 
                     description, test_val, input_real, tanh_out, output_real);
        end
    endtask
    
    // Main test
    initial begin
        $display("=== Simple Tanh Testbench ===\n");
        
        // Linear region (0 ? x < 0.25)
        $display("Linear Region:");
        test(16'h0000, "Zero        ");
        test(16'h0020, "0.125       ");
        test(16'h003F, "0.246       ");
        
        // LUT region (0.25 ? x ? 3.0) 
        $display("\nLUT Region:");
        test(16'h0040, "0.25        ");
        test(16'h0100, "1.0         ");
        test(16'h0200, "2.0         ");
        test(16'h0300, "3.0         ");
        
        // Saturation (x > 3.0)
        $display("\nPositive Saturation:");
        test(16'h0400, "4.0         ");
        test(16'h7FFF, "Max Pos     ");
        
        // Negative linear (-0.25 < x < 0)
        $display("\nNegative Linear:");
        test(16'hFFE0, "-0.125      ");
        test(16'hFFC1, "-0.246      ");
        
        // Negative LUT (-3.0 ? x ? -0.25)
        $display("\nNegative LUT:");
        test(16'hFFC0, "-0.25       ");
        test(16'hFF00, "-1.0        ");
        test(16'hFE00, "-2.0        ");
        test(16'hFD00, "-3.0        ");
        
        // Negative saturation (x < -3.0)
        $display("\nNegative Saturation:");
        test(16'hFC00, "-4.0        ");
        test(16'h8000, "Max Neg     ");
        
        // Boundaries
        $display("\nBoundary Tests:");
        test(16'h003E, "Just < 0.25 ");
        test(16'h0041, "Just > 0.25 ");
        test(16'h02FF, "Just < 3.0  ");
        test(16'h0301, "Just > 3.0  ");
        
        $display("\n=== Test Complete ===");
        $finish;
    end
    
endmodule