// Simple Testbench for Fixed-Point Subtractor
`timescale 1ns / 1ps

module tb_sub_simple();

    // Test signals
    reg [11:0] a, b;
    wire [11:0] diff;
    wire overflow;
    
    // DUT instantiation
    sub_fixed dut (
        .a(a),
        .b(b),
        .diff(diff),
        .overflow(overflow)
    );
    
    // Test procedure
    initial begin
        $display("Fixed-Point Subtractor - Simple Test");
        $display("A\t\tB\t\tA-B\t\tOVF");
        $display("---\t\t---\t\t---\t\t---");
        
        // Test 1: Basic positive subtraction
        a = 12'h0A0; // 2.5
        b = 12'h050; // 1.25
        #10;
        $display("%h\t\t%h\t\t%h\t\t%b", a, b, diff, overflow);
        
        // Test 2: Negative result
        a = 12'h040; // 1.0
        b = 12'h080; // 2.0
        #10;
        $display("%h\t\t%h\t\t%h\t\t%b", a, b, diff, overflow);
        
        // Test 3: Both negative
        a = 12'h8C0; // -3.0
        b = 12'h860; // -1.5
        #10;
        $display("%h\t\t%h\t\t%h\t\t%b", a, b, diff, overflow);
        
        // Test 4: Mixed signs (pos - neg = pos + pos)
        a = 12'h080; // 2.0
        b = 12'h880; // -2.0
        #10;
        $display("%h\t\t%h\t\t%h\t\t%b", a, b, diff, overflow);
        
        // Test 5: Zero result
        a = 12'h0C0; // 3.0
        b = 12'h0C0; // 3.0
        #10;
        $display("%h\t\t%h\t\t%h\t\t%b", a, b, diff, overflow);
        
        // Test 6: Zero operand
        a = 12'h000; // 0.0
        b = 12'h040; // 1.0
        #10;
        $display("%h\t\t%h\t\t%h\t\t%b", a, b, diff, overflow);
        
        $display("Test completed");
        $finish;
    end
    
endmodule