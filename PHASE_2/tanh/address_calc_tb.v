module tb_tanh_addr_calculator;
    
    reg [15:0] input_value;
    wire [8:0] lut_addr;
    wire addr_valid, use_symmetry, saturate_low, saturate_high;
    
    // Instantiate DUT
    tanh_addr_calculator_no_mult dut (
        .input_value(input_value),
        .lut_addr(lut_addr),
        .addr_valid(addr_valid),
        .use_symmetry(use_symmetry),
        .saturate_low(saturate_low),
        .saturate_high(saturate_high)
    );
    
    // Test stimulus
    initial begin
        $display("Testing Tanh Address Calculator");
        $display("Input_Hex | Input_Dec | Address | Valid | Symm | Expected_Range");
        $display("----------|-----------|---------|-------|------|---------------");
        
        // Test key values
        test_input(16'h0040, "0.25 (min)");     // Should give address 0
        test_input(16'h0080, "0.50");           // Should give address ~25
        test_input(16'h0100, "1.00");           // Should give address ~75
        test_input(16'h0180, "1.50");           // Should give address ~125
        test_input(16'h0200, "2.00");           // Should give address ~175
        test_input(16'h0280, "2.50");           // Should give address ~225
        test_input(16'h0300, "3.00 (max)");     // Should give address 275
        
        // Test negative values
        $display("\nNegative Values:");
        test_input(16'hFF00, "-1.00");          // Should use symmetry
        test_input(16'hFE00, "-2.00");          // Should use symmetry
        
        // Test saturation
        $display("\nSaturation Tests:");
        test_input(16'h0020, "0.125 (low)");    // Should saturate low
        test_input(16'h0400, "4.00 (high)");    // Should saturate high
        
        $finish;
    end
    
    task test_input;
        input [15:0] value;
        input [127:0] description;
        begin
            input_value = value;
            #1; // Wait for combinational logic
            $display("0x%04X    | %-12s | %7d | %5b | %4b | %s", 
                     value, description, lut_addr, addr_valid, use_symmetry, description);
        end
    endtask
    
endmodule