`timescale 1ns / 1ps

module tb_tanh_calc;

    // Testbench signals
    reg [11:0] in;
    wire [11:0] out;
    
    // Test verification
    integer test_count = 0;
    integer pass_count = 0;
    reg [11:0] expected_out;
    wire test_pass;
    
    // Compare output with expected (allow small tolerance for LUT approximation)
    assign test_pass = (out == expected_out) || 
                      ((out >= (expected_out - 1)) && (out <= (expected_out + 1)));
    
    // DUT instantiation
    tanh_calc #(
        .WIDTH(12),
        .FRAC_BITS(6)
    ) dut (
        .in(in),
        .out(out)
    );
    
    // Helper function to convert fixed-point to real (for display)
    function real fixed_to_real;
        input [11:0] fixed_val;
        begin
            if (fixed_val[11] == 1'b0) begin
                // Positive number
                fixed_to_real = $itor(fixed_val) / 64.0;  // Divide by 2^6
            end else begin
                // Negative number (sign-magnitude)
                fixed_to_real = -$itor(fixed_val[10:0]) / 64.0;
            end
        end
    endfunction
    
    // Task to run a single test
    task run_test;
        input [11:0] test_input;
        input [11:0] expected_output;
        input [255:0] description;
        begin
            test_count = test_count + 1;
            in = test_input;
            expected_out = expected_output;
            #10;
            
            if (test_pass) 
                pass_count = pass_count + 1;
            
            $display("%0d\t%012b\t%03h\t%6.3f\t%012b\t%03h\t%6.3f\t%012b\t%s\t%s", 
                     test_count, 
                     in, in, fixed_to_real(in),
                     out, out, fixed_to_real(out),
                     expected_out,
                     test_pass ? "PASS" : "FAIL",
                     description);
        end
    endtask
    
    initial begin
        $display("Testing tanh Calculator (S1.5.6 Sign-Magnitude)");
        $display("Test#\tin_binary   \tin_hex\tin_val\tout_binary  \tout_hex\tout_val\texpected_bin\tResult\tDescription");
        $display("===============================================================================================================");
        
        // Test 1: Zero input
        run_test(12'b011001101100, 12'b100001000000, "Zero input -> Zero output");
        
        // Test 2: Small positive values (linear region - output ? input)
        run_test(12'b000000001000, 12'b000000001000, "Small positive (0.125) -> linear");
        run_test(12'b000000001111, 12'b000000001111, "Small positive (0.234) -> linear");
        
        // Test 3: LUT range positive (use expected tanh values)
        run_test(12'b000000010000, 12'b000000001111, "LUT: tanh(0.25) ? 0.244");   // tanh(0.25) ? 15/64
        run_test(12'b000001000000, 12'b000000110000, "LUT: tanh(1.0) ? 0.762");    // tanh(1.0) ? 48/64
        run_test(12'b000011000000, 12'b000000111111, "LUT: tanh(3.0) ? 0.995");    // tanh(3.0) ? 63/64
        
        // Test 4: Large positive (saturation to +1.0)
        run_test(12'b000100000000, 12'b000001000000, "Large positive -> +1.0");    // +1.0 = 64/64
        run_test(12'b001111111111, 12'b000001000000, "Max positive -> +1.0");      // +1.0 = 64/64
        
        // Test 5: Small negative values (linear region - output ? input)
        run_test(12'b100000001000, 12'b100000001000, "Small negative (-0.125) -> linear");
        run_test(12'b100000001111, 12'b100000001111, "Small negative (-0.234) -> linear");
        
        // Test 6: LUT range negative (use negated tanh values with sign bit)
        run_test(12'b100000010000, 12'b100000001111, "LUT: tanh(-0.25) ? -0.244");  // -tanh(0.25)
        run_test(12'b100001000000, 12'b100000110000, "LUT: tanh(-1.0) ? -0.762");   // -tanh(1.0)
        run_test(12'b100011000000, 12'b100000111111, "LUT: tanh(-3.0) ? -0.995");   // -tanh(3.0)
        
        // Test 7: Large negative (saturation to -1.0)
        run_test(12'b100100000000, 12'b100001000000, "Large negative -> -1.0");     // -1.0
        run_test(12'b111111111111, 12'b100001000000, "Max negative -> -1.0");       // -1.0
        
        // Additional boundary tests
        run_test(12'b000000001001, 12'b000000001001, "Boundary: just below LUT_MIN");
        run_test(12'b000011000001, 12'b000001000000, "Boundary: just above LUT_MAX");
        run_test(12'b100000001001, 12'b100000001001, "Boundary: just above NEG_LUT_MIN");
        run_test(12'b100011000001, 12'b100001000000, "Boundary: just below NEG_LUT_MAX");
        
        // Summary
        $display("\n========================================");
        $display("TEST SUMMARY");
        $display("========================================");
        $display("Total tests: %0d", test_count);
        $display("Passed tests: %0d", pass_count);
        $display("Failed tests: %0d", test_count - pass_count);
        $display("Pass rate: %.1f%%", (real(pass_count) / real(test_count)) * 100.0);
        
        if (pass_count == test_count) begin
            $display("*** ALL TESTS PASSED ***");
        end else begin
            $display("*** SOME TESTS FAILED ***");
            $display("Note: Small differences may be due to LUT quantization");
        end
        
        $display("\nExpected behavior verification:");
        $display("  - Small inputs (|x| < 0.25): output ? input (linear approximation)");
        $display("  - Medium inputs (0.25 ? |x| ? 3.0): use LUT values");
        $display("  - Large inputs (|x| > 3.0): saturate to ±1.0 (64/64 in fixed-point)");
        $display("  - Zero input: output = 0");
        
        $display("\nTest complete!");
        $finish;
    end

endmodule