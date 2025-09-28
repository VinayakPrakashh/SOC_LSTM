`timescale 1ns / 1ps

module tb_fixed_point_comparator_simple;

    // Testbench signals
    reg [11:0] a, b;
    wire a_gt_b, a_lt_b, a_eq_b, a_gte_b, a_lte_b;
    
    // Test verification signals
    reg expected_gt, expected_lt, expected_eq;
    wire test_pass;
    integer test_count = 0;
    integer pass_count = 0;
    
    // Check if output matches expected
    assign test_pass = (a_gt_b == expected_gt) && (a_lt_b == expected_lt) && (a_eq_b == expected_eq);
    
    // DUT instantiation
    fixed_point_comparator dut (
        .a(a),
        .b(b),
        .a_gt_b(a_gt_b),
        .a_lt_b(a_lt_b),
        .a_eq_b(a_eq_b),
        .a_gte_b(a_gte_b),
        .a_lte_b(a_lte_b)
    );
    
    initial begin
        $display("Testing Fixed-Point Comparator (S1.5.6 Sign-Magnitude)");
        $display("Test#\ta_hex\tb_hex\tgt\tlt\teq\tpass");
        
        // Test 1: Equal values
        test_count = 1;
        a = 12'h000; b = 12'h000; expected_gt = 0; expected_lt = 0; expected_eq = 1; #10;
        if (test_pass) pass_count = pass_count + 1;
        $display("%0d\t%03h\t%03h\t%b\t%b\t%b\t%b", test_count, a, b, a_gt_b, a_lt_b, a_eq_b, test_pass);
        
        test_count = 2;
        a = 12'h040; b = 12'h040; expected_gt = 0; expected_lt = 0; expected_eq = 1; #10;  // 1.0 == 1.0
        if (test_pass) pass_count = pass_count + 1;
        $display("%0d\t%03h\t%03h\t%b\t%b\t%b\t%b", test_count, a, b, a_gt_b, a_lt_b, a_eq_b, test_pass);
        
        test_count = 3;
        a = 12'h840; b = 12'h840; expected_gt = 0; expected_lt = 0; expected_eq = 1; #10;  // -1.0 == -1.0
        if (test_pass) pass_count = pass_count + 1;
        $display("%0d\t%03h\t%03h\t%b\t%b\t%b\t%b", test_count, a, b, a_gt_b, a_lt_b, a_eq_b, test_pass);
        
        // Test 2: Positive vs Positive
        test_count = 4;
        a = 12'h040; b = 12'h020; expected_gt = 1; expected_lt = 0; expected_eq = 0; #10;  // 1.0 > 0.5
        if (test_pass) pass_count = pass_count + 1;
        $display("%0d\t%03h\t%03h\t%b\t%b\t%b\t%b", test_count, a, b, a_gt_b, a_lt_b, a_eq_b, test_pass);
        
        test_count = 5;
        a = 12'h020; b = 12'h040; expected_gt = 0; expected_lt = 1; expected_eq = 0; #10;  // 0.5 < 1.0
        if (test_pass) pass_count = pass_count + 1;
        $display("%0d\t%03h\t%03h\t%b\t%b\t%b\t%b", test_count, a, b, a_gt_b, a_lt_b, a_eq_b, test_pass);
        
        // Test 3: Negative vs Negative
        test_count = 6;
        a = 12'h840; b = 12'h880; expected_gt = 1; expected_lt = 0; expected_eq = 0; #10;  // -1.0 > -2.0
        if (test_pass) pass_count = pass_count + 1;
        $display("%0d\t%03h\t%03h\t%b\t%b\t%b\t%b", test_count, a, b, a_gt_b, a_lt_b, a_eq_b, test_pass);
        
        test_count = 7;
        a = 12'h880; b = 12'h840; expected_gt = 0; expected_lt = 1; expected_eq = 0; #10;  // -2.0 < -1.0
        if (test_pass) pass_count = pass_count + 1;
        $display("%0d\t%03h\t%03h\t%b\t%b\t%b\t%b", test_count, a, b, a_gt_b, a_lt_b, a_eq_b, test_pass);
        
        // Test 4: Positive vs Negative
        test_count = 8;
        a = 12'h010; b = 12'h810; expected_gt = 1; expected_lt = 0; expected_eq = 0; #10;  // 0.25 > -0.25
        if (test_pass) pass_count = pass_count + 1;
        $display("%0d\t%03h\t%03h\t%b\t%b\t%b\t%b", test_count, a, b, a_gt_b, a_lt_b, a_eq_b, test_pass);
        
        test_count = 9;
        a = 12'h040; b = 12'h840; expected_gt = 1; expected_lt = 0; expected_eq = 0; #10;  // 1.0 > -1.0
        if (test_pass) pass_count = pass_count + 1;
        $display("%0d\t%03h\t%03h\t%b\t%b\t%b\t%b", test_count, a, b, a_gt_b, a_lt_b, a_eq_b, test_pass);
        
        // Test 5: Zero comparisons
        test_count = 10;
        a = 12'h000; b = 12'h010; expected_gt = 0; expected_lt = 1; expected_eq = 0; #10;  // 0 < 0.25
        if (test_pass) pass_count = pass_count + 1;
        $display("%0d\t%03h\t%03h\t%b\t%b\t%b\t%b", test_count, a, b, a_gt_b, a_lt_b, a_eq_b, test_pass);
        
        test_count = 11;
        a = 12'h000; b = 12'h810; expected_gt = 1; expected_lt = 0; expected_eq = 0; #10;  // 0 > -0.25
        if (test_pass) pass_count = pass_count + 1;
        $display("%0d\t%03h\t%03h\t%b\t%b\t%b\t%b", test_count, a, b, a_gt_b, a_lt_b, a_eq_b, test_pass);
        
        // Test 6: Boundary values for tanh
        test_count = 12;
        a = 12'h010; b = 12'h0C0; expected_gt = 0; expected_lt = 1; expected_eq = 0; #10;  // 0.25 < 3.0
        if (test_pass) pass_count = pass_count + 1;
        $display("%0d\t%03h\t%03h\t%b\t%b\t%b\t%b", test_count, a, b, a_gt_b, a_lt_b, a_eq_b, test_pass);
        
        test_count = 13;
        a = 12'h810; b = 12'h8C0; expected_gt = 1; expected_lt = 0; expected_eq = 0; #10;  // -0.25 > -3.0
        if (test_pass) pass_count = pass_count + 1;
        $display("%0d\t%03h\t%03h\t%b\t%b\t%b\t%b", test_count, a, b, a_gt_b, a_lt_b, a_eq_b, test_pass);
        
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
        end
        
        $display("Test complete!");
        $finish;
    end

endmodule