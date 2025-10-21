module tb_address_calc;

    // Testbench signals
    reg [15:0] input_value;
    wire [8:0] lut_addr;
    wire addr_valid;
    wire use_symmetry;
    wire saturate_low;
    wire saturate_high;

    // Instantiate DUT
    tanh_addr_calculator dut (
        .input_value(input_value),
        .lut_addr(lut_addr),
        .addr_valid(addr_valid),
        .use_symmetry(use_symmetry),
        .saturate_low(saturate_low),
        .saturate_high(saturate_high)
    );

    // Test stimulus
    initial begin
        $display("Tanh Address Calculator Test");
        $display("============================");
        $display("Input_Hex | Input_Dec | LUT_Addr | Valid | Symm | Sat_L | Sat_H | Expected_Input");
        $display("----------|-----------|----------|-------|------|-------|-------|---------------");
        
        // Test cases with expected results
        test_input(16'h0040, "0.25");    // Minimum input
        test_input(16'h0060, "0.375");   
        test_input(16'h0080, "0.50");    
        test_input(16'h00A0, "0.625");   
        test_input(16'h00C0, "0.75");    
        test_input(16'h0100, "1.00");    // 1.0
        test_input(16'h0140, "1.25");    
        test_input(16'h0180, "1.50");    
        test_input(16'h01C0, "1.75");    
        test_input(16'h0200, "2.00");    // 2.0
        test_input(16'h0280, "2.50");    
        test_input(16'h0300, "3.00");    // Maximum input
        
        $display("\nNegative Input Tests:");
        $display("----------|-----------|----------|-------|------|-------|-------|---------------");
        
        // Negative values (two's complement)
        test_input(16'hFF00, "-1.00");   // -1.0
        test_input(16'hFE80, "-1.50");   // -1.5
        test_input(16'hFE00, "-2.00");   // -2.0
        
        $display("\nSaturation Tests:");
        $display("----------|-----------|----------|-------|------|-------|-------|---------------");
        
        // Saturation cases
        test_input(16'h0020, "0.125");   // Below minimum
        test_input(16'h0010, "0.0625");  // Well below minimum
        test_input(16'h0400, "4.00");    // Above maximum
        test_input(16'h0500, "5.00");    // Well above maximum
        
        $finish;
    end

    task test_input;
        input [15:0] value;
        input [63:0] description;
        begin
            input_value = value;
            #1; // Small delay for combinational logic
            $display("0x%04X    | %-9s | %8d | %5b | %4b | %5b | %5b | %s", 
                     value, description, lut_addr, addr_valid, use_symmetry, 
                     saturate_low, saturate_high, description);
        end
    endtask

endmodule